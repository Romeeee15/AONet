#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from options.base_options import BaseOptions
warnings.filterwarnings("ignore", category=UserWarning)
from HiFDenoise import HiF

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class CAconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CAconv, self).__init__()
        self.ca = ChannelAttention(in_planes=in_channels)
        self.conv = DoubleConv2D(in_channels=in_channels, out_channels=out_channels)
    
    def forward(self, x):
        y = self.ca(x)
        return self.conv(x * y)


class Conv2D(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,  kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DoubleConv2D(nn.Module):
    """(convolution=> ReLU) * 2"""
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16,out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(16,out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels,  kernel_size=3, padding=1),
            nn.GroupNorm(16,out_channels),
            nn.PReLU(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2D(channels,channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
    
class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, channels,  bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2)  #cyr6e# channels ?

        self.conv = DoubleConv2D(channels*2,channels)
        self.channel_attention =ChannelAttention(64*2)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        attention_value = self.channel_attention(x)
        out = x.mul(attention_value)

        return self.conv(out)

class UNet(nn.Module):
    def __init__(self, in_channels,channels, n_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.channels=channels
        self.n_classes = n_classes
        self.inc = DoubleConv2D(in_channels, channels)
        self.down1 = Down(channels)
        self.down2 = Down(channels)
        self.down3 = Down(channels)
        self.down4 = Down(channels)
        self.up1 = Up(channels)
        self.up2 = Up(channels)
        self.up3 = Up(channels)
        self.up4 = Up(channels)
        self.outc = Conv2D(channels, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        self.low_feat = x4
        
        x4 = self.up1(x5, x4)
        x3 = self.up2(x4, x3)
        x2 = self.up3(x3, x2)
        feature = self.up4(x2, x1)
        logits = self.outc(feature)
        return logits
    
    def get_low_feature(self):
        return self.low_feat

class VoluS(nn.Module):
    def __init__(self, in_channels,out_channels,h):
        super().__init__()

        self.conv1 =nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=h),
            nn.GroupNorm(h, out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, groups=h),
            nn.GroupNorm(h, out_channels),
            nn.PReLU(out_channels),
        )
        self.conv2 = CAconv(in_channels, out_channels)
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels*2+in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(h, out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(h, out_channels),
            nn.PReLU(out_channels),
        )
    def forward(self, x, HF):
        
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2, HF], dim=1)
        x = self.conv3(x)
        return x

class AONet(nn.Module):
    def __init__(self, in_channels,n_classes):
        super(AONet, self).__init__()
        self.dhfM = HiF()
        self.VoluS1 = VoluS(in_channels, 256, h=32)   
        self.VoluS2 = VoluS(256, 128, h=32)
        self.VoluS3 = VoluS(128, 64, h=32)
        self.SegNet2D = UNet(64, 64, n_classes)
    
    def forward(self, x):
        HF = self.dhfM(x)
        x = self.VoluS1(x, HF[0])
        x = self.VoluS2(x, HF[1])
        x = self.VoluS3(x, HF[2])
        x = self.SegNet2D(x)
        return x
    
    def get_low_feature(self):
        return self.SegNet2D.get_low_feature()

if __name__ == "__main__":
    ac = AONet(320, 5).cuda()
    x = torch.rand(2,320,304,304).cuda()
    y = ac(x)
    print(y.shape)
