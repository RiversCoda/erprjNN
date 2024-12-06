# dn_dataloader.py
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dn_utils import load_mat, normalize, lowpass_filter

class DenoisingDataset(Dataset):
    def __init__(self, noise_path_pattern, raw_path_pattern, transform=None, 
                 normalize_method='-1_to_1', apply_filter=False, mean=0, std=1):
        self.noise_files = sorted(glob.glob(noise_path_pattern))
        self.raw_files = sorted(glob.glob(raw_path_pattern))
        self.transform = transform
        self.normalize_method = normalize_method
        self.apply_filter = apply_filter
        self.mean = mean
        self.std = std
        assert len(self.noise_files) == len(self.raw_files), "Mismatch between noise and raw files."

    def __len__(self):
        return len(self.noise_files)

    def __getitem__(self, idx):
        noise_file = self.noise_files[idx]
        raw_file = self.raw_files[idx]

        noise_data = load_mat(noise_file)[1]  # Index 1
        raw_data = load_mat(raw_file)[1]      # Index 1

        if self.apply_filter:
            noise_data = lowpass_filter(noise_data)
            raw_data = lowpass_filter(raw_data)

        noise_data = normalize(noise_data, method=self.normalize_method, mean=self.mean, std=self.std)
        raw_data = normalize(raw_data, method=self.normalize_method, mean=self.mean, std=self.std)

        # 修改这里：在最后一个维度上扩展
        noise_tensor = torch.tensor(noise_data, dtype=torch.float32).unsqueeze(-1)  # Shape: (2000, 1)
        raw_tensor = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(-1)      # Shape: (2000, 1)

        if self.transform:
            noise_tensor = self.transform(noise_tensor)
            raw_tensor = self.transform(raw_tensor)

        return noise_tensor, raw_tensor

def get_dataloader(noise_path_pattern, raw_path_pattern, batch_size=32, shuffle=True, 
                  num_workers=4, normalize_method='-1_to_1', apply_filter=False, mean=0, std=1):
    dataset = DenoisingDataset(noise_path_pattern, raw_path_pattern, 
                               normalize_method=normalize_method, 
                               apply_filter=apply_filter, 
                               mean=mean, std=std)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                            num_workers=num_workers, pin_memory=True)
    return dataloader
