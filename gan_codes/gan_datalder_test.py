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

        print("开始加载数据...")
        total_files = 0
        total_segments = 0

        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                print(f"错误：数据目录 {data_dir} 不存在")
                continue

            mat_files = glob.glob(os.path.join(data_dir, '*.mat'))
            print(f"在目录 {data_dir} 中找到 {len(mat_files)} 个 .mat 文件")
            total_files += len(mat_files)

            if len(mat_files) == 0:
                print(f"警告：目录 {data_dir} 中没有 .mat 文件")
                continue

            for file_path in mat_files:
                try:
                    mat = scipy.io.loadmat(file_path)
                    # 检查 'accresult' 键是否存在
                    if 'accresult' not in mat:
                        print(f"警告：文件 {file_path} 中没有 'accresult' 键")
                        continue

                    accresult = mat['accresult']
                    print(f"文件 {file_path} 中 'accresult' 形状：{accresult.shape}")

                    # 根据实际数据结构调整索引
                    if accresult.shape == (4, 5000):
                        # 假设第二行是 Y 轴数据
                        signal = accresult[1, :]
                    else:
                        print(f"警告：文件 {file_path} 中的 'accresult' 形状不符合预期")
                        continue

                    # 检查信号长度
                    if len(signal) < self.window_size:
                        print(f"警告：文件 {file_path} 中的信号长度不足，无法进行窗口分割")
                        continue

                    # 预处理信号
                    if self.preprocess:
                        signal = self._preprocess_signal(signal)

                    # 归一化信号
                    if self.normalize:
                        std = np.std(signal)
                        if std == 0:
                            print(f"警告：文件 {file_path} 中的信号标准差为0，无法归一化")
                            continue
                        signal = (signal - np.mean(signal)) / std

                    # 滑动窗口分割
                    segments = self._segment_signal(signal)
                    num_segments = len(segments)
                    print(f"文件 {file_path} 生成了 {num_segments} 个片段")
                    total_segments += num_segments

                    if num_segments == 0:
                        print(f"警告：文件 {file_path} 没有生成任何片段")
                        continue

                    self.data.extend(segments)

                    # 分配标签
                    if noise_type:
                        self.labels.extend([noise_type] * num_segments)
                    else:
                        self.labels.extend(['clean'] * num_segments)

                except Exception as e:
                    print(f"错误：加载文件 {file_path} 时发生异常：{e}")

        print(f"数据加载完成，共加载了 {total_files} 个文件，生成了 {total_segments} 个片段")
        print(f"数据集大小：{len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]
        # 如果需要，将标签转换为数值
        return torch.FloatTensor(signal), label

    def _preprocess_signal(self, signal):
        # 添加您的预处理步骤，例如滤波
        return signal

    def _segment_signal(self, signal):
        segments = []
        for start in range(0, len(signal) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            segments.append(signal[start:end])
        return segments

def get_dataloader(data_dirs, batch_size=32, shuffle=True, **kwargs):
    dataset = HeartbeatDataset(data_dirs, **kwargs)
    dataset_length = len(dataset)
    print(f"创建 DataLoader，数据集长度：{dataset_length}")
    if dataset_length == 0:
        print("错误：数据集为空，无法创建 DataLoader")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
