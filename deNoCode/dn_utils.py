import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

def save_model(model, epoch, loss, model_name):
    os.makedirs('saved_models', exist_ok=True)
    model_filename = f'{model_name}_epoch{epoch}_loss{loss:.4f}.pth'
    model_path = os.path.join('saved_models', model_filename)
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f'Model loaded from {model_path}')

def low_pass_filter(data, cutoff=100, fs=1000, order=5):
    """
    对数据应用低通滤波器。

    参数：
    - data: 输入的时间序列数据。
    - cutoff: 截止频率，默认 100Hz。
    - fs: 采样率，根据您的数据设置，默认 1000Hz。
    - order: 滤波器的阶数，默认 5。

    返回：
    - 滤波后的数据。
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y.astype(np.float32)

def normalize(data):
    """
    对数据进行标准化处理，使其均值为 0，标准差为 1。

    参数：
    - data: 输入的时间序列数据。

    返回：
    - 标准化后的数据。
    """
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data.astype(np.float32)

# -1 到 1的标准化
def normalize_1_1(data):
    aa = 1