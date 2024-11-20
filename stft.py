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
nperseg = 64     # 每段的长度，增大可提高频率分辨率
noverlap = nperseg // 2  # 重叠长度，可根据需要调整
nfft = 128      # FFT点数，增大可提高频率分辨率
fmin = 0         # 最低频率，单位Hz
fmax = 100       # 最高频率，单位Hz

# 获取目录下所有.mat文件的列表
mat_files = [f for f in os.listdir(data_path) if f.endswith('.mat')]

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

    # 创建绘图窗口，包含两个子图
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # 绘制1D信号
    axs[0].plot(time, signal)
    axs[0].set_title(f'{mat_file}的原始信号')
    axs[0].set_ylabel('振幅')
    axs[0].set_xlabel('时间 [秒]')
    axs[0].grid(True)

    # 绘制频谱图
    pcm = axs[1].pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    axs[1].set_title(f'{mat_file}的STFT幅值谱')
    axs[1].set_ylabel('频率 [Hz]')
    axs[1].set_xlabel('时间 [秒]')
    axs[1].set_ylim([fmin, fmax])
    fig.colorbar(pcm, ax=axs[1], label='幅值')

    plt.tight_layout()
    plt.show()
