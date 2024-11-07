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
                # Extract y-axis acceleration data (1 x 5000)
                signal = mat['accresult'][0, 0, 1, :]  # Adjust indices if necessary

                # Preprocess signal
                if self.preprocess:
                    signal = self._preprocess_signal(signal)

                # Normalize signal
                if self.normalize:
                    signal = (signal - np.mean(signal)) / np.std(signal)

                # Sliding window
                segments = self._segment_signal(signal)
                self.data.extend(segments)

                # Assign labels
                if noise_type:
                    self.labels.extend([noise_type] * len(segments))
                else:
                    self.labels.extend(['clean'] * len(segments))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]
        # Convert label to numerical value if necessary
        return torch.FloatTensor(signal), label

    def _preprocess_signal(self, signal):
        # Add your preprocessing steps here (e.g., filtering)
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
