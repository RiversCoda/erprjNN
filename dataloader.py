import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

class HeartbeatDataset(Dataset):
    def __init__(self, data_dir, mode='train', val_ratio=0):
        self.data_dir = data_dir
        self.mode = mode
        self.window_size = 2000
        self.stride = 500
        self.users = []
        self.data = []
        self.labels = []
        self._load_data()
        if self.mode == 'train':
            self._split_data(val_ratio)
        elif self.mode == 'val':
            self._split_data(val_ratio, val_only=True)

    def _load_data(self):
        for username in os.listdir(self.data_dir):
            user_path = os.path.join(self.data_dir, username, 'scg')
            if not os.path.isdir(user_path):
                continue
            user_signals = []
            for file in os.listdir(user_path):
                if file.endswith('.mat'):
                    mat = loadmat(os.path.join(user_path, file))
                    if 'accresult' in mat:
                        signal = mat['accresult'][1, :]
                    elif 'scg_data' in mat:
                        signal = mat['scg_data'][1, :]

                    # signal = mat['accresult'][1, :]  # 使用索引1的信号
                    # 切片信号
                    slices = self._slice_signal(signal)
                    user_signals.extend(slices)
            # 规范化用户的信号
            user_signals = self._normalize(np.array(user_signals))
            self.data.extend(user_signals)
            self.labels.extend([username] * len(user_signals))
            self.users.append(username)

    def _slice_signal(self, signal):
        slices = []
        for start in range(0, len(signal) - self.window_size + 1, self.stride):
            end = start + self.window_size
            slices.append(signal[start:end])
        return slices

    def _normalize(self, data):
        # 最小-最大规范化
        min_vals = data.min(axis=1, keepdims=True)
        max_vals = data.max(axis=1, keepdims=True)
        normalized = (data - min_vals) / (max_vals - min_vals + 1e-8)
        return normalized

    def _split_data(self, val_ratio, val_only=False):
        # 从训练数据中划分验证集
        total_samples = len(self.data)
        val_size = int(total_samples * val_ratio)
        if val_only:
            self.data = self.data[:val_size]
            self.labels = self.labels[:val_size]
        else:
            self.val_data = self.data[:val_size]
            self.val_labels = self.labels[:val_size]
            self.data = self.data[val_size:]
            self.labels = self.labels[val_size:]

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'val':
            anchor = self.data[idx]
            anchor_label = self.labels[idx]
    
            # Get positive sample
            pos_indices = [i for i, label in enumerate(self.labels) if label == anchor_label and i != idx]
            pos_idx = np.random.choice(pos_indices)
            positive = self.data[pos_idx]
    
            # Get negative sample
            neg_indices = [i for i, label in enumerate(self.labels) if label != anchor_label]
            
            if len(neg_indices) == 0:
                raise ValueError("No negative samples available for anchor with label:", anchor_label)
            
            neg_idx = np.random.choice(neg_indices)
            negative = self.data[neg_idx]
    
            return (
                torch.FloatTensor(anchor),
                torch.FloatTensor(positive),
                torch.FloatTensor(negative)
            )

        elif self.mode == 'test':
            sample = self.data[idx]
            label = self.labels[idx]
            return torch.FloatTensor(sample), label
def get_dataloaders(data_dir, batch_size, val_ratio=0.1, num_workers=0, pin_memory=False):
    # 创建数据集实例
    train_dataset = HeartbeatDataset(data_dir, mode='train', val_ratio=val_ratio)
    val_dataset = HeartbeatDataset(data_dir, mode='val', val_ratio=val_ratio)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, val_loader
