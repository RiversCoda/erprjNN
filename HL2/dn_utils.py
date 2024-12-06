# dn_utils.py
import numpy as np
import scipy.io
from scipy.signal import butter, lfilter, stft
import os

def load_mat(file_path):
    mat = scipy.io.loadmat(file_path)
    return mat['accresult']

def normalize(data, method='-1_to_1', mean=0, std=1):
    if method == '-1_to_1':
        data_min = data.min()
        data_max = data.max()
        normalized = 2 * (data - data_min) / (data_max - data_min) - 1
    elif method == 'z-score':
        normalized = (data - mean) / std
    else:
        raise ValueError("Unsupported normalization method.")
    return normalized

def lowpass_filter(data, cutoff=100, fs=500, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

def compute_stft(data, nperseg=64, noverlap=32, nfft=512):
    f, t, Zxx = stft(data, fs=500, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    return Zxx

def save_mat(file_path, data, var_name='accresult'):
    scipy.io.savemat(file_path, {var_name: data})
