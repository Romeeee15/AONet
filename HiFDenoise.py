#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from options.base_options import BaseOptions
warnings.filterwarnings("ignore", category=UserWarning)

class WaveletTransform(nn.Module):
    def __init__(self, wavelet='db2', eps=1e-8):
        super().__init__()
        self.opt = BaseOptions().parse()
        self.wavelet = wavelet.lower()
        self.eps = eps
        for name in [
            'h0a_col', 'h1a_col', 'h0s_col', 'h1s_col',
            'h0a_row', 'h1a_row', 'h0s_row', 'h1s_row'
        ]:
            self.register_buffer(name, None, persistent=False)

    @staticmethod
    def _get_filters(wavelet, device, dtype):
        w = wavelet.lower()
        if w == 'haar':
            inv_sqrt2 = 1.0 / math.sqrt(2.0)
            h0a = torch.tensor([inv_sqrt2,  inv_sqrt2], device=device, dtype=dtype)
            h1a = torch.tensor([-inv_sqrt2, inv_sqrt2], device=device, dtype=dtype)
            h0s = torch.tensor([inv_sqrt2,  inv_sqrt2], device=device, dtype=dtype)
            h1s = torch.tensor([inv_sqrt2, -inv_sqrt2], device=device, dtype=dtype)
        elif w == 'db2':
            sqrt3 = math.sqrt(3.0)
            denom = 4.0 * math.sqrt(2.0)
            h0a = torch.tensor([
                (1 + sqrt3) / denom,
                (3 + sqrt3) / denom,
                (3 - sqrt3) / denom,
                (1 - sqrt3) / denom
            ], device=device, dtype=dtype)
            h1a = torch.tensor([ h0a[-1], -h0a[-2],  h0a[-3], -h0a[-4] ], device=device, dtype=dtype)
            h0s = torch.flip(h0a, dims=[0]).clone()
            h1s = torch.flip(h1a, dims=[0]).clone()
        elif w == 'cdf53':
            h0a = torch.tensor([0.0, -1/8, 1/4, 3/4, 1/4, -1/8, 0.0], device=device, dtype=dtype)
            h1a = torch.tensor([0.0, -1/2, 1.0, -1/2, 0.0], device=device, dtype=dtype)
            h0s = torch.tensor([0.0,  1/2, 1.0,  1/2, 0.0], device=device, dtype=dtype)
            h1s = torch.tensor([0.0,  1/8, -1/4, 3/4, -1/4, 1/8, 0.0], device=device, dtype=dtype)
        else:
            raise ValueError(f"Unsupported wavelet type: {wavelet}")
        return h0a, h1a, h0s, h1s

    def _ensure_filters(self, x):
        if self.h0a_col is not None and self.h0a_col.device == x.device and self.h0a_col.dtype == x.dtype:
            return
        h0a, h1a, h0s, h1s = self._get_filters(self.wavelet, x.device, x.dtype)

        k_a = h0a.numel()
        k_s = h0s.numel()
        self.h0a_col = h0a.view(1, 1, k_a, 1)
        self.h1a_col = h1a.view(1, 1, k_a, 1)
        self.h0s_col = h0s.view(1, 1, k_s, 1)
        self.h1s_col = h1s.view(1, 1, k_s, 1)
        self.h0a_row = h0a.view(1, 1, 1, k_a)
        self.h1a_row = h1a.view(1, 1, 1, k_a)
        self.h0s_row = h0s.view(1, 1, 1, k_s)
        self.h1s_row = h1s.view(1, 1, 1, k_s)

    @staticmethod
    def _reflect_pad_1d(x, pad, dim):
        if pad <= 0:
            return x
        if dim == -2:  # H
            return F.pad(x, (0, 0, pad, pad), mode='reflect')
        elif dim == -1:  # W
            return F.pad(x, (pad, pad, 0, 0), mode='reflect')
        else:
            raise ValueError("dim must be -2 (H) or -1 (W)")

    def forward(self, x):
        B, C, H, W = x.shape
        self._ensure_filters(x)

        k_a_h = self.h0a_col.shape[-2]
        k_a_w = self.h0a_row.shape[-1]

        x_pad_h = self._reflect_pad_1d(x, (k_a_h - 1) // 2, dim=-2)
        Lo = F.conv2d(x_pad_h, self.h0a_col.expand(C, 1, -1, -1), stride=(2, 1), groups=C)
        Hi = F.conv2d(x_pad_h, self.h1a_col.expand(C, 1, -1, -1), stride=(2, 1), groups=C)

        Lo = self._reflect_pad_1d(Lo, (k_a_w - 1) // 2, dim=-1)
        Hi = self._reflect_pad_1d(Hi, (k_a_w - 1) // 2, dim=-1)
        LL = F.conv2d(Lo, self.h0a_row.expand(C, 1, -1, -1), stride=(1, 2), groups=C)

        LF, HF = self.reconstruct(x, LL)


        return HF

    def _idwt2(self, LL, LH, HL, HH, target_size=None):
        assert LL.shape == LH.shape == HL.shape == HH.shape
        B, C, H2, W2 = LL.shape
        self._ensure_filters(LL)

        LL_up = F.conv_transpose2d(LL, self.h0s_row.expand(C, 1, -1, -1), stride=(1, 2), groups=C)
        LH_up = F.conv_transpose2d(LH, self.h1s_row.expand(C, 1, -1, -1), stride=(1, 2), groups=C)
        HL_up = F.conv_transpose2d(HL, self.h0s_row.expand(C, 1, -1, -1), stride=(1, 2), groups=C)
        HH_up = F.conv_transpose2d(HH, self.h1s_row.expand(C, 1, -1, -1), stride=(1, 2), groups=C)
        Lo = LL_up + LH_up
        Hi = HL_up + HH_up
        
        x_lo = F.conv_transpose2d(Lo, self.h0s_col.expand(C, 1, -1, -1), stride=(2, 1), groups=C)
        x_hi = F.conv_transpose2d(Hi, self.h1s_col.expand(C, 1, -1, -1), stride=(2, 1), groups=C)
        x = x_lo + x_hi

        if target_size is not None:
            Ht, Wt = target_size
            x = x[..., :Ht, :Wt]
        return x

    def inverse(self, LL, LH, HL, HH, original_size=None):
        x_LL = self._idwt2(LL, torch.zeros_like(LH), torch.zeros_like(HL), torch.zeros_like(HH), target_size=original_size)
        x_LH = self._idwt2(LL, LH, torch.zeros_like(HL), torch.zeros_like(HH), target_size=original_size)
        x_HL = self._idwt2(LL, torch.zeros_like(LH), HL, torch.zeros_like(HH), target_size=original_size)
        x_HH = self._idwt2(LL, torch.zeros_like(LH), torch.zeros_like(HL), HH, target_size=original_size)

        return x_LL, x_LH, x_HL, x_HH

    def reconstruct(self, x, LL, original_size=None):
        reconstructed_image = self._idwt2(LL, torch.zeros_like(LL), torch.zeros_like(LL), torch.zeros_like(LL), target_size=original_size)
        LF = reconstructed_image[:, :, :self.opt.data_size[1], :self.opt.data_size[2]]
        HF = x-LF
        return LF, HF


class AdaDConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, groups=1, use_channel=True, use_nin=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = (kernel_size-1) // 2
        self.stride = stride
        self.in_channels = in_channels
        self.groups = groups
        self.use_channel = use_channel

        if use_nin:
            mid_channel = min((kernel_size ** 2 // 2), 4)
            self.weight_net = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=groups * mid_channel , stride=stride,
                        kernel_size=kernel_size, bias=False, padding=self.pad, groups=groups),
                nn.BatchNorm2d(self.groups * mid_channel), 
                nn.ReLU(True),
                nn.Conv2d(in_channels=groups * mid_channel, out_channels=groups * kernel_size ** 2, stride=1,
                        kernel_size=1, bias=False, padding=0, groups=groups),
                nn.BatchNorm2d(self.groups * kernel_size ** 2), 
            )

        else:
            self.weight_net = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=groups * kernel_size ** 2, stride=stride,
                        kernel_size=kernel_size, bias=False, padding=self.pad, groups=groups),
                nn.BatchNorm2d(self.groups * kernel_size ** 2), 
            )

        if use_channel:
            self.channel_net = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), 
                nn.Conv2d(in_channels=in_channels, out_channels= in_channels // 4, kernel_size=1, bias=False),
                nn.ReLU(True),
                nn.Conv2d(in_channels=in_channels // 4, out_channels = in_channels, kernel_size=1, bias=False),
            )


    def forward(self, x):
        b, c, h, w = x.shape
        oh = (h - 1) // self.stride + 1
        ow = (w - 1) // self.stride + 1
        weight = self.weight_net(x) 
        _weight = weight
        weight = weight.reshape(b, self.groups, 1, self.kernel_size ** 2, oh, ow) 
        weight = weight.repeat(1, 1, c // self.groups, 1, 1, 1)

        if self.use_channel:
            tmp = self.channel_net(x).reshape(b, self.groups, c // self.groups, 1, 1, 1)
            weight = weight * tmp
        weight = weight.permute(0, 1, 2, 4, 5, 3).softmax(dim=-1)
        weight = weight.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)

        pad_x = F.pad(x, pad=[self.pad] * 4, mode='reflect')
        pad_x = pad_x.unfold(2, self.kernel_size,self.stride).unfold(3, self.kernel_size, self.stride)
        pad_x = pad_x.reshape(b, self.groups, c // self.groups, oh, ow, self.kernel_size, self.kernel_size)
        res = weight * pad_x
        res = res.sum(dim=(-1, -2)).reshape(b, c, oh, ow)
        return res
    
class HiF(nn.Module):
    def __init__(self, channels=[320, 256, 128]):
        super().__init__()
        self.wave = WaveletTransform()
        self.denoise = AdaDConv(in_channels=channels[0])
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels[0],channels[1], kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels[0],channels[2], kernel_size=1),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        x1 =  self.denoise(self.wave(x))
        x2 = self.conv1(x1)
        x3 = self.conv2(x1)
        return [x1, x2, x3]