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

def lowpass_filter(data, cutoff=100, fs=1000, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)