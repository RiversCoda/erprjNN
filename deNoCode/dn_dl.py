import os
import glob
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dn_utils import *
from sklearn.preprocessing import StandardScaler  # 添加用于标准化的数据包

class DenoiseDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.noise_dir = os.path.join(root_dir, mode, 'noise')
        self.raw_dir = os.path.join(root_dir, mode, 'raw')
        self.noisy_files = glob.glob(os.path.join(self.noise_dir, '*', 'scg', '*.mat'))
        self.raw_files = glob.glob(os.path.join(self.raw_dir, '*', 'scg', '*.mat'))
        self.data_pairs = self._create_data_pairs()
        self.scaler = StandardScaler()  # 初始化标准化对象

    def _create_data_pairs(self):
        data_pairs = []
        raw_dict = {}
        for raw_file in self.raw_files:
            key = self._get_key(raw_file)
            raw_dict[key] = raw_file

        for noisy_file in self.noisy_files:
            key = self._get_key(noisy_file).rsplit('_noise', 1)[0]
            if key in raw_dict:
                data_pairs.append((noisy_file, raw_dict[key]))
        return data_pairs

    def _get_key(self, filepath):
        return os.path.splitext(os.path.basename(filepath))[0]

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        noisy_file, raw_file = self.data_pairs[idx]
        noisy_mat = sio.loadmat(noisy_file)['accresult'][1]
        raw_mat = sio.loadmat(raw_file)['accresult'][1]
        
        # 将数据转换为float32类型的numpy数组
        noisy_data = np.array(noisy_mat, dtype=np.float32)
        raw_data = np.array(raw_mat, dtype=np.float32)
        
        # 应用低通滤波器
        noisy_data = lowpass_filter(noisy_data)
        raw_data = lowpass_filter(raw_data)
        
        # 数据标准化
        noisy_data = self.scaler.fit_transform(noisy_data.reshape(-1, 1)).flatten()
        raw_data = self.scaler.transform(raw_data.reshape(-1, 1)).flatten()
        
        return noisy_data, raw_data

def get_dataloader(root_dir, mode='train', batch_size=32, num_workers=4):
    dataset = DenoiseDataset(root_dir, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
