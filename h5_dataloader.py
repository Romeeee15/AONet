#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import random
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import math
import cv2
import json
from PIL import Image
from scipy.ndimage import convolve, gaussian_laplace, white_tophat, sobel, gaussian_filter
from skimage.filters import frangi

def create_gaussian_high_pass_filter(shape, cutoff, sigma=5):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2 
    

    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
    

    gaussian_low_pass = np.exp(-(distance**2) / (2 * sigma**2))
    gaussian_high_pass = 1 - gaussian_low_pass
    

    gaussian_high_pass = gaussian_high_pass / gaussian_high_pass.max()
    
    return gaussian_high_pass


def apply_fourier_filter(image_stack, filter_stack):

    N, H, W = image_stack.shape
    filtered_stack = np.zeros_like(image_stack, dtype=np.complex64)
    
    for i in range(N):

        f_transform = fftshift(fft2(image_stack[i]))
        

        filtered_f = f_transform * filter_stack[i]
        

        filtered_image = ifft2(ifftshift(filtered_f))
        

        filtered_stack[i] = np.real(filtered_image)
    
    return filtered_stack


def fft2(image):

    return np.fft.fft2(image)

def ifft2(image):

    return np.fft.ifft2(image)

def fftshift(image):

    return np.fft.fftshift(image)

def ifftshift(image):

    return np.fft.ifftshift(image)


class CubeDataset(Dataset):
    def __init__(self, data_dir, data_id, modality, labelity, data_clip, is_dataaug=True, use_sam=False, use_filter=False):
        super().__init__()
        self.is_dataaug = is_dataaug
        with open(data_clip, 'r') as f: self.data_clip = json.load(f) #path2data
        self.data = h5py.File(f'{data_dir}/{modality[0]}.h5', 'r')
        self.label = h5py.File(f'{data_dir}/{labelity[0]}.h5', 'r')
        self.item_list = list(self.data.keys())[data_id[0]: data_id[1]]
        self.use_sam = use_sam
        self.use_filter = use_filter
        
    def __getitem__(self, index):
        item_name = self.item_list[index]
        item_data = self.data[item_name][...]
        clip = self.data_clip[item_name]
        item_data=item_data.transpose(1,2,0)[clip[0]:clip[1]]
        item_label = self.label[item_name][...]
        if self.is_dataaug:
            item_data, item_label = self.augmentation(item_data, item_label)

        
        
        if self.use_filter:
            item_data_slice = self.data_slice(item_data, interval=10, new_c=320)
            item_data_fre = self.get_enhanced_single_channel(item_data)
            item_data = np.concatenate([item_data_slice, item_data_fre], axis=0) 
        
        if not self.use_sam:
            item_data = torch.from_numpy(np.ascontiguousarray(item_data))
            item_label = torch.from_numpy(np.ascontiguousarray(item_label[None]))
            return item_data, item_label, item_name
        
        else:
            image = np.sum(item_data, axis=0)
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            sam_data = image.transpose(2, 0, 1)
            
            item_data = torch.from_numpy(np.ascontiguousarray(item_data))
            item_label = torch.from_numpy(np.ascontiguousarray(item_label[None]))
            print(f'data {(item_data).shape}')
            return item_data, item_label, item_name



    def __len__(self):
        return len(self.item_list)


    def get_unsharp_mask_3d(self, item_data, sigma=1.5, strength=1.0):
        img = item_data.astype(np.float32)
        low_freq = gaussian_filter(img, sigma=sigma)
        high_freq = img - low_freq
        enhanced = img + strength * high_freq
        
        return np.clip(enhanced, img.min(), img.max())


    def get_enhanced_single_channel(self, item_data, weight=0.5):
        img = item_data.astype(np.float32)
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        high_freq = gaussian_laplace(img_norm, sigma=1.0)
        bright_details = white_tophat(img_norm, size=3)
        enhanced = img_norm + weight * np.abs(high_freq) + weight * bright_details
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min() + 1e-8)
        return enhanced
    
    
    
    
    
    
    
    def get_3d_tophat(self, item_data, size=3):
        structure = np.ones((size, size, size))
        tophat_feat = white_tophat(item_data.astype(np.float32), footprint=structure)
        return tophat_feat
    
    def get_3d_sobel_magnitude(self, item_data):

        img = item_data.astype(np.float32)

        dx = sobel(img, axis=1)
        dy = sobel(img, axis=2)
        dz = sobel(img, axis=0)
        
        magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
        return magnitude


    def get_3d_log(self, item_data, sigma=1.0):


        log_feat = gaussian_laplace(item_data.astype(np.float32), sigma=sigma)
        return log_feat


    def compute_3d_frangi(self, item_data):
        vesselness = frangi(item_data, sigmas=[1, 2], black_ridges=False)
        return vesselness.astype(np.float32)

    

    def get_3d_laplacian(self, item_data):

        kernel = np.array([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ])
        

        high_freq = convolve(item_data.astype(np.float32), kernel, mode='mirror')
        return high_freq

    def to_image(self, item_data, img_name, subname):
        
        img = item_data.mean(axis=0)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = Image.fromarray(img, mode='L')
        filename = f"./delete/{img_name}_{subname}.png"
        img.save(filename)
        
        return
    def data_slice(self, item_data, interval=10, new_c=320):

        c = item_data.shape[0]
        step = math.floor(c / new_c)  

        slices = []
        for i in range(new_c):
            start = int(i * step)  
            end = start + interval  
            if end > c:  
                start = c - interval
                end = c
            slices.append(item_data[start:end, :, :])

        compressed_slices = [s.mean(axis=0, keepdims=True) for s in slices]


        return np.concatenate(compressed_slices, axis=0)
    
    def apply_filter(self, data, gaussian_cutoff=15, gaussian_sigma=10):
        
        item_data = torch.from_numpy(np.ascontiguousarray(data[None]))
        noisy_stack = np.squeeze(item_data)

        gaussian_high_pass_filters = np.array([create_gaussian_high_pass_filter(img.shape, gaussian_cutoff, gaussian_sigma) for img in noisy_stack])

        
        filtered_gaussian_high_stack = apply_fourier_filter(noisy_stack, gaussian_high_pass_filters)
            
        item_data = filtered_gaussian_high_stack.real
        return item_data
    
    def augmentation(self, image, annotation):
        if random.randint(0, 4) == 0:
            image = np.flip(image, axis=1).copy()
            annotation = np.flip(annotation, axis=0).copy()

        if random.randint(0, 4) == 0:
            image = np.flip(image, axis=2).copy()
            annotation = np.flip(annotation, axis=1).copy()

        if random.randint(0, 4) == 0:
            k = random.randint(1, 4)
            image = np.rot90(image, k, axes=(1, 2)).copy()
            annotation = np.rot90(annotation, k, axes=(0, 1)).copy()

        if random.randint(0, 4) == 0:
            dis = random.randint(1, 20)
            if random.randint(0, 1) == 0:
                image = np.concatenate((image[-dis:, :, :], image[:-dis, :, :]), axis=0)
            else:
                image = np.concatenate((image[dis:, :, :], image[:dis, :, :]), axis=0)
            image = np.ascontiguousarray(image)

        image = np.ascontiguousarray(image, dtype=np.float32)
        annotation = np.ascontiguousarray(annotation)

        return image, annotation
