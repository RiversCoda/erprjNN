import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter, stft
import pywt

# 加载数据
data = scipy.io.loadmat('collect_data/device3/noise_phone/sjx/scg/10-30-13-56-43.mat')
accresult = data['accresult']
signal = accresult[1, :]  # 提取第二条信号

# 时间轴
fs = 1000  # 假设采样率为1000Hz，如有实际值请替换
t = np.arange(len(signal)) / fs

# STFT参数
stft_nperseg = 256
stft_noverlap = 128
stft_freq_range = (0, 100)  # 显示频率区间为0到100Hz

# 1. 带通滤波器
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

filtered_signal = bandpass_filter(signal, 6, 40, fs)

# 2. 小波去噪
def wavelet_denoising(data, wavelet='db8', level=1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeffs[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeffs[1:])
    return pywt.waverec(coeffs, wavelet)

wavelet_signal = wavelet_denoising(signal)

# 3. Savitzky-Golay滤波器
savgol_signal = savgol_filter(signal, window_length=101, polyorder=3)

# 绘制结果
plt.figure(figsize=(15, 15))

# 绘制原始信号
plt.subplot(4, 2, 1)
plt.plot(t, signal)
plt.title('原始信号')

# 原始信号的STFT
f, t_stft, Zxx = stft(signal, fs, nperseg=stft_nperseg, noverlap=stft_noverlap)
plt.subplot(4, 2, 2)
plt.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
plt.ylim(stft_freq_range)
plt.colorbar()
plt.title('原始信号的STFT')

# 绘制带通滤波后信号
plt.subplot(4, 2, 3)
plt.plot(t, filtered_signal)
plt.title('带通滤波后信号')

# 带通滤波后信号的STFT
f, t_stft, Zxx = stft(filtered_signal, fs, nperseg=stft_nperseg, noverlap=stft_noverlap)
plt.subplot(4, 2, 4)
plt.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
plt.ylim(stft_freq_range)
plt.colorbar()
plt.title('带通滤波后信号的STFT')

# 绘制小波去噪后信号
plt.subplot(4, 2, 5)
plt.plot(t, wavelet_signal)
plt.title('小波去噪后信号')

# 小波去噪后信号的STFT
f, t_stft, Zxx = stft(wavelet_signal, fs, nperseg=stft_nperseg, noverlap=stft_noverlap)
plt.subplot(4, 2, 6)
plt.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
plt.ylim(stft_freq_range)
plt.colorbar()
plt.title('小波去噪后信号的STFT')

# 绘制Savitzky-Golay滤波后信号
plt.subplot(4, 2, 7)
plt.plot(t, savgol_signal)
plt.title('Savitzky-Golay滤波后信号')

# Savitzky-Golay滤波后信号的STFT
f, t_stft, Zxx = stft(savgol_signal, fs, nperseg=stft_nperseg, noverlap=stft_noverlap)
plt.subplot(4, 2, 8)
plt.pcolormesh(t_stft, f, np.abs(Zxx), shading='gouraud')
plt.ylim(stft_freq_range)
plt.colorbar()
plt.title('Savitzky-Golay滤波后信号的STFT')

plt.tight_layout()
plt.show()
