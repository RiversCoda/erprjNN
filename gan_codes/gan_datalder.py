import os
import glob
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class HeartbeatDataset(Dataset):
    def __init__(self, data_dirs, noise_type=None, window_size=2000, step_size=200, normalize=True, preprocess=True):
        self.data = []
        self.labels = []
        self.noise_type = noise_type
        self.window_size = window_size
        self.step_size = step_size
        self.normalize = normalize
        self.preprocess = preprocess

        for data_dir in data_dirs:
            mat_files = glob.glob(os.path.join(data_dir, '*.mat'))
            for file_path in mat_files:
                mat = scipy.io.loadmat(file_path)
                # 调试信息
                print(f"Processing file: {file_path}")
                print(f"Keys in mat: {mat.keys()}")
                print(f"Type of mat['accresult']: {type(mat['accresult'])}")
                print(f"Shape of mat['accresult']: {mat['accresult'].shape}")

                # 根据实际形状调整索引
                accresult = mat['accresult']  # 形状为 (4, 5000)
                # 假设第二行是 y 轴加速度数据
                signal = accresult[1, :]  # 提取第 1 行的数据

                # 预处理信号
                if self.preprocess:
                    signal = self._preprocess_signal(signal)

                # 归一化信号
                if self.normalize:
                    signal = (signal - np.mean(signal)) / np.std(signal)

                # 滑动窗口分割
                segments = self._segment_signal(signal)
                self.data.extend(segments)

                # 分配标签
                if self.noise_type:
                    self.labels.extend([self.noise_type] * len(segments))
                else:
                    self.labels.extend(['clean'] * len(segments))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]
        return torch.FloatTensor(signal), label

    def _preprocess_signal(self, signal):
        # 在这里添加您的预处理步骤（例如滤波）
        return signal

    def _segment_signal(self, signal):
        segments = []
        for start in range(0, len(signal) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            segments.append(signal[start:end])
        return segments

def get_dataloader(data_dirs, batch_size=32, shuffle=True, **kwargs):
    dataset = HeartbeatDataset(data_dirs, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
