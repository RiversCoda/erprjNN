# dn_utils.py

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, lfilter, stft, istft
import torch

def load_mat_file(filepath):
    data = loadmat(filepath)
    return data['accresult'][1, :]  # 使用第二条通道的数据

def normalize(data, method='minmax', mean=0, std=1):
    if method == 'minmax': # 0 to 1 normalization
        data_min = data.min()
        data_max = data.max()
        return (data - data_min) / (data_max - data_min)
    elif method == 'standard':
        return (data - mean) / std
    
    elif method == 'minmax_minus1_1': # -1 to 1 normalization
        data_min = data.min()
        data_max = data.max()
        return 2 * ((data - data_min) / (data_max - data_min)) - 1
    else:
        raise ValueError("Unsupported normalization method.")

def butter_lowpass_filter(data, cutoff=50, fs=500, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def compute_stft(data, nperseg=64, noverlap=32, nfft=512):
    f, t, Zxx = stft(data, fs=500, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    return f, t, Zxx

def compute_istft(Zxx, nperseg=64, noverlap=32, nfft=512):
    _, x = istft(Zxx, fs=500, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    return x

def stft_to_tensor(Zxx, mode='magnitude'):
    if mode == 'magnitude':
        spectrogram = np.abs(Zxx)
    elif mode == 'magnitude_phase':
        spectrogram = np.concatenate((np.abs(Zxx), np.angle(Zxx)), axis=0)
    elif mode == 'real_imag':
        spectrogram = np.concatenate((Zxx.real, Zxx.imag), axis=0)
    else:
        raise ValueError("Unsupported STFT mode.")
    return torch.tensor(spectrogram, dtype=torch.float32)

def tensor_to_stft(tensor, mode='magnitude_phase'):
    if mode == 'magnitude':
        magnitude = tensor.numpy()
        phase = np.zeros_like(magnitude)
    elif mode == 'magnitude_phase':
        magnitude = tensor.numpy()[:tensor.size(0)//2]
        phase = tensor.numpy()[tensor.size(0)//2:]
    elif mode == 'real_imag':
        real = tensor.numpy()[:tensor.size(0)//2]
        imag = tensor.numpy()[tensor.size(0)//2:]
        magnitude = real + 1j * imag
    else:
        raise ValueError("Unsupported tensor to STFT mode.")
    return magnitude, phase

def save_params_log(params, filepath):
    with open(filepath, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
