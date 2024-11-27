import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import stft

# 定义包含.mat文件的目录路径
data_path = r'collect_data\device3\\p4-test\sjx\scg'

# 可调整的参数
fs = 1000        # 采样频率，单位Hz，根据实际情况调整
window = 'hann'  # 窗函数类型，可选'hann'、'hamming'等
nperseg =  64    # 每段的长度，增大可提高频率分辨率
noverlap = nperseg // 2  # 重叠长度，可根据需要调整
nfft = 512     # FFT点数，增大可提高频率分辨率
fmin = 0         # 最低频率，单位Hz
fmax = 200       # 最高频率，单位Hz
SCALE_NUM = 128  # 小波尺度数

# 获取目录下所有.mat文件的列表
mat_files = [f for f in os.listdir(data_path) if f.endswith('.mat')]

# 定义Morlet小波函数
def morlet_wavelet(scale, time, f0= 64):
    """生成Morlet小波，scale为尺度，time为时间轴"""
    t = time / scale
    wavelet = np.exp(-0.5 * t**2) * np.cos(2 * np.pi * f0 * t)
    return wavelet

# 手动实现CWT（连续小波变换）
def cwt(signal, scales, fs):
    """手动实现连续小波变换"""
    time = np.arange(len(signal)) / fs  # 时间轴，信号的每个采样点对应的时间值
    cwt_result = np.zeros((len(scales), len(signal)), dtype=complex)
    for i, scale in enumerate(scales):
        wavelet = morlet_wavelet(scale, time)  
        cwt_result[i, :] = np.convolve(signal, wavelet, mode='same')
    return cwt_result, time

for mat_file in mat_files:
    # 加载.mat文件
    mat_data = loadmat(os.path.join(data_path, mat_file))
    # 提取'accresult'变量
    accresult = mat_data['accresult']  # 应为形状(4, 5000)
    # 选择第二条数据（索引为1）
    signal = accresult[1, :]
    signal = -signal  # 反转信号
    # 计算时间轴
    time = np.arange(len(signal)) / fs

    # 计算STFT
    f, t, Zxx = stft(signal, fs=fs, window=window, nperseg=nperseg,
                     noverlap=noverlap, nfft=nfft)
    # 限制频率范围为0-100 Hz
    freq_mask = (f >= fmin) & (f <= fmax)
    f = f[freq_mask]
    Zxx = Zxx[freq_mask, :]

    # 手动进行CWT（连续小波变换）
    scales = np.arange(1, SCALE_NUM)  # 小波的尺度
    cwt_result, time_cwt = cwt(signal, scales, fs)
    freqs_cwt = fs / scales  # 对应的频率

    # 创建绘图窗口，包含四个子图
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))

    # 绘制1D信号
    axs[0].plot(time, signal)
    axs[0].set_title(f'{mat_file}的原始信号')
    axs[0].set_ylabel('振幅')
    axs[0].set_xlabel('时间 [秒]')
    axs[0].grid(True)

    # 绘制STFT频谱图
    pcm = axs[1].pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    axs[1].set_title(f'{mat_file}的STFT幅值谱')
    axs[1].set_ylabel('频率 [Hz]')
    axs[1].set_xlabel('时间 [秒]')
    axs[1].set_ylim([fmin, fmax])
    fig.colorbar(pcm, ax=axs[1], label='幅值')

    # 绘制CWT频谱图
    axs[2].imshow(np.abs(cwt_result), aspect='auto', extent=[0, time[-1], freqs_cwt[-1], freqs_cwt[0]], cmap='jet')
    axs[2].set_title(f'{mat_file}的小波变换幅值谱')
    axs[2].set_ylabel('频率 [Hz]')
    axs[2].set_xlabel('时间 [秒]')
    fig.colorbar(axs[2].images[0], ax=axs[2], label='幅值')

    # 绘制STFT频谱图与1D信号对比
    # 标准化signal。使其为0-200之间
    signal = (signal - signal.min()) / (signal.max() - signal.min()) * 200
    axs[3].plot(time, signal, label="1D Signal", color='white', alpha=0.6)
    axs[3].pcolormesh(t, f, np.abs(Zxx), shading='gouraud', alpha=0.9)
    axs[3].set_title(f'{mat_file}的STFT频谱图与原始信号对比')
    axs[3].set_ylabel('频率 [Hz]')
    axs[3].set_xlabel('时间 [秒]')
    axs[3].legend(loc='upper right')

    # 让1D信号的纵轴自适应
    axs[3].set_ylim([signal.min(), signal.max()])

    plt.tight_layout()
    plt.show()
