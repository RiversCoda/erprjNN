import os
import glob
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dn_utils import *

Ncutoff = 70

class DenoiseDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.noise_dir = os.path.join(root_dir, mode, 'noise')
        self.raw_dir = os.path.join(root_dir, mode, 'raw')
        self.noisy_files = glob.glob(os.path.join(self.noise_dir, '*', 'scg', '*.mat'))
        self.raw_files = glob.glob(os.path.join(self.raw_dir, '*', 'scg', '*.mat'))
        self.data_pairs = self._create_data_pairs()

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
        noisy_data = np.array(noisy_mat, dtype=np.float32)
        raw_data = np.array(raw_mat, dtype=np.float32)

        # 应用 N Hz 低通滤波器
        noisy_data = low_pass_filter(noisy_data, cutoff=Ncutoff, fs=1000, order=5)
        raw_data = low_pass_filter(raw_data, cutoff=Ncutoff, fs=1000, order=5)

        # 应用归一化
        noisy_data = normalize(noisy_data)
        raw_data = normalize(raw_data)

        return noisy_data, raw_data


def get_dataloader(root_dir, mode='train', batch_size=32, num_workers=4):
    dataset = DenoiseDataset(root_dir, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

