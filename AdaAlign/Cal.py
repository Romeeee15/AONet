import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks
import cv2
import json

class CubeDataset(Dataset):
    def __init__(self, data_dir, modality, labelity):
        super().__init__()
        self.data = h5py.File(f'{data_dir}/{modality[0]}.h5', 'r')
        self.label = h5py.File(f'{data_dir}/{labelity[0]}.h5', 'r')
        self.item_list = list(self.data.keys())

    def __getitem__(self, index):
        item_name = self.item_list[index]
        item_data = self.data[item_name][...]
        item_label = self.label[item_name][...]
        item_data=item_data.transpose(1,2,0)
        return item_data, item_label, item_name

    def __len__(self):
        return len(self.item_list)

if __name__ == '__main__':
    data_dir = ''
    modality = ['OCTA_6mm']
    labelity = ['GT_CAVF']
    dataset = CubeDataset(data_dir, modality, labelity)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    SII_list = []
    for i, (sample) in enumerate(dataloader):
        data, label, name =  sample
        data = data[0]
        print(f'the {i+1} sample is {name}')
        SII_value = torch.fft.fft2(data).abs().mean(dim=(1,2)).tolist()
        SII_list.append({"name": name[0], "SII": SII_value})
    with open('.../SII_6mm.json', 'w') as f:
        json.dump(SII_list, f, indent=4)
