import os
import glob
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class HeartbeatDatasetCycleGAN(Dataset):
    def __init__(self, data_dirs_A, data_dirs_B, window_size=2000, step_size=200, normalize=True, preprocess=True):
        self.data_A = []
        self.data_B = []
        self.window_size = window_size
        self.step_size = step_size
        self.normalize = normalize
        self.preprocess = preprocess

        # 加载域A的数据（干净的信号）
        for data_dir in data_dirs_A:
            mat_files = glob.glob(os.path.join(data_dir, '*.mat'))
            for file_path in mat_files:
                mat = scipy.io.loadmat(file_path)
                accresult = mat['accresult']  # 假设形状为 (4, N)
                signal = accresult[1, :]  # 提取第二行
                if self.preprocess:
                    signal = self._preprocess_signal(signal)
                if self.normalize:
                    signal = (signal - np.mean(signal)) / np.std(signal)
                segments = self._segment_signal(signal)
                self.data_A.extend(segments)

        # 加载域B的数据（含噪声的信号）
        for data_dir in data_dirs_B:
            mat_files = glob.glob(os.path.join(data_dir, '*.mat'))
            for file_path in mat_files:
                mat = scipy.io.loadmat(file_path)
                accresult = mat['accresult']
                signal = accresult[1, :]
                if self.preprocess:
                    signal = self._preprocess_signal(signal)
                if self.normalize:
                    signal = (signal - np.mean(signal)) / np.std(signal)
                segments = self._segment_signal(signal)
                self.data_B.extend(segments)

    def __len__(self):
        return max(len(self.data_A), len(self.data_B))

    def __getitem__(self, idx):
        idx_A = idx % len(self.data_A)
        idx_B = idx % len(self.data_B)
        signal_A = self.data_A[idx_A]
        signal_B = self.data_B[idx_B]
        return torch.FloatTensor(signal_A), torch.FloatTensor(signal_B)

    def _preprocess_signal(self, signal):
        # 在这里添加您的预处理步骤（例如滤波）
        return signal

    def _segment_signal(self, signal):
        segments = []
        for start in range(0, len(signal) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            segments.append(signal[start:end])
        return segments

def get_dataloader_cyclegan(data_dirs_A, data_dirs_B, batch_size=32, shuffle=True, **kwargs):
    dataset = HeartbeatDatasetCycleGAN(data_dirs_A, data_dirs_B, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# 在 cyclegan_dataloader.py 中添加以下代码

class HeartbeatDatasetTest(Dataset):
    def __init__(self, data_dirs_B, window_size=2000, step_size=200, normalize=True, preprocess=True):
        self.data_B = []
        self.window_size = window_size
        self.step_size = step_size
        self.normalize = normalize
        self.preprocess = preprocess

        # 加载域B的数据（含噪声的信号）
        for data_dir in data_dirs_B:
            mat_files = glob.glob(os.path.join(data_dir, '*.mat'))
            for file_path in mat_files:
                mat = scipy.io.loadmat(file_path)
                accresult = mat['accresult']
                signal = accresult[1, :]
                if self.preprocess:
                    signal = self._preprocess_signal(signal)
                if self.normalize:
                    signal = (signal - np.mean(signal)) / np.std(signal)
                segments = self._segment_signal(signal)
                self.data_B.extend(segments)

    def __len__(self):
        return len(self.data_B)

    def __getitem__(self, idx):
        signal_B = self.data_B[idx]
        return torch.FloatTensor(signal_B)

    def _preprocess_signal(self, signal):
        # 在这里添加您的预处理步骤（例如滤波）
        return signal

    def _segment_signal(self, signal):
        segments = []
        for start in range(0, len(signal) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            segments.append(signal[start:end])
        return segments

def get_test_dataloader(data_dirs_B, batch_size=1, shuffle=False, **kwargs):
    dataset = HeartbeatDatasetTest(data_dirs_B, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
