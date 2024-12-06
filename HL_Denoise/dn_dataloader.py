# dn_dataloader.py

import os
import glob
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
from dn_utils import load_mat_file, normalize, butter_lowpass_filter

class DenoisingDataset(Dataset):
    def __init__(self, clean_paths, noise_paths_dict, transform=None, normalize_method='minmax', mean=0, std=1, apply_filter=False):
        """
        clean_paths: list of clean .mat file paths
        noise_paths_dict: dict mapping clean file to list of corresponding noise file paths
        """
        self.clean_paths = clean_paths
        self.noise_paths_dict = noise_paths_dict
        self.transform = transform
        self.normalize_method = normalize_method
        self.mean = mean
        self.std = std
        self.apply_filter = apply_filter

    def __len__(self):
        return sum(len(noise_list) for noise_list in self.noise_paths_dict.values())

    def __getitem__(self, idx):
        # Find which clean file and which noise file corresponds to the idx
        cumulative = 0
        for clean_path, noise_list in self.noise_paths_dict.items():
            if idx < cumulative + len(noise_list):
                noise_idx = idx - cumulative
                noise_path = noise_list[noise_idx]
                clean_data = load_mat_file(clean_path)
                noise_data = load_mat_file(noise_path)
                # 输入为安静数据和噪音数据的叠加
                input_data = noise_data + clean_data
                target_data = clean_data

                # 可选滤波
                if self.apply_filter:
                    input_data = butter_lowpass_filter(input_data)
                    target_data = butter_lowpass_filter(target_data)

                # 标准化
                input_data = normalize(input_data, method=self.normalize_method, mean=self.mean, std=self.std)
                target_data = normalize(target_data, method=self.normalize_method, mean=self.mean, std=self.std)

                # 转换为float32
                input_data = input_data.astype(np.float32)
                target_data = target_data.astype(np.float32)

                if self.transform:
                    input_data = self.transform(input_data)
                    target_data = self.transform(target_data)

                return input_data, target_data
            cumulative += len(noise_list)
        raise IndexError

def get_dataloader(clean_dir, noise_dir, batch_size=32, shuffle=True, num_workers=4, normalize_method='minmax', mean=0, std=1, apply_filter=False):
    clean_paths = glob.glob(os.path.join(clean_dir, '*', 'scg', '*.mat'))
    noise_paths_dict = {}
    for clean_path in clean_paths:
        parts = clean_path.split(os.sep)
        subject = parts[-3]  # 假设路径格式固定
        filename = os.path.basename(clean_path).replace('.mat', '')
        noise_pattern = os.path.join(noise_dir, subject, 'scg', f"{filename}_noise*.mat")
        noise_files = glob.glob(noise_pattern)
        noise_files.sort()
        noise_paths_dict[clean_path] = noise_files

    dataset = DenoisingDataset(clean_paths, noise_paths_dict, normalize_method=normalize_method, mean=mean, std=std, apply_filter=apply_filter)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, len(dataset)

def get_test_dataloader(clean_dir, noise_dir, batch_size=32, shuffle=False, num_workers=4, normalize_method='minmax', mean=0, std=1, apply_filter=False):
    return get_dataloader(clean_dir, noise_dir, batch_size, shuffle, num_workers, normalize_method, mean, std, apply_filter)
