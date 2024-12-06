import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import stft
import matplotlib

# 设置字体以支持中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 定义Morlet小波函数
def morlet_wavelet(scale, time, f0=64):
    """
    生成Morlet小波
    :param scale: 小波尺度
    :param time: 时间轴
    :param f0: 小波的中心频率（默认64Hz）
    :return: 生成的Morlet小波
    """
    t = time / scale
    wavelet = np.exp(-0.5 * t**2) * np.cos(2 * np.pi * f0 * t)
    return wavelet

# 手动实现CWT（连续小波变换）
def cwt(signal, scales, fs):
    """
    计算信号的连续小波变换
    :param signal: 输入信号
    :param scales: 小波尺度范围
    :param fs: 采样频率
    :return: 连续小波变换结果和时间轴
    """
    time = np.arange(len(signal)) / fs
    cwt_result = np.zeros((len(scales), len(signal)), dtype=complex)
    for i, scale in enumerate(scales):
        wavelet = morlet_wavelet(scale, time)
        cwt_result[i, :] = np.convolve(signal, wavelet, mode='same')
    return cwt_result, time

# 数据读取函数
def read_data(data_path):
    """
    从指定路径读取.mat文件并返回文件列表和数据
    :param data_path: 数据文件所在目录路径
    :return: 文件名和数据的列表
    """
    mat_files = [f for f in os.listdir(data_path) if f.endswith('.mat')]
    data = []
    for mat_file in mat_files:
        mat_data = loadmat(os.path.join(data_path, mat_file))
        accresult = mat_data['accresult']
        data.append((mat_file, accresult))
    return data

# 数据处理与绘图函数
def process_and_plot_signal(signal, 
                            time, 
                            fs, 
                            scales, 
                            filename='1Ddata', 
                            ONOFF_1D=1, 
                            ONOFF_STFT=1, 
                            ONOFF_CWT=0, 
                            ONOFF_STFT_1D=1, 
                            window='hann', 
                            nperseg=64, 
                            noverlap=None, 
                            nfft=512, 
                            fmin=0, 
                            fmax=100):
    """
    处理信号并进行绘图
    :param signal: 输入信号
    :param time: 时间轴
    :param fs: 采样频率
    :param scales: 小波尺度范围
    :param filename: 文件名（用于标题）
    :param ONOFF_1D: 是否绘制原始信号（布尔值）
    :param ONOFF_STFT: 是否绘制STFT图（布尔值）
    :param ONOFF_CWT: 是否绘制CWT图（布尔值）
    :param ONOFF_STFT_1D: 是否绘制STFT与1D信号对比图（布尔值）
    :param window: STFT的窗函数类型
    :param nperseg: STFT每段的长度
    :param noverlap: STFT重叠长度
    :param nfft: FFT点数
    :param fmin: STFT最小频率
    :param fmax: STFT最大频率
    """
    if noverlap is None:
        noverlap = nperseg // 2

    # STFT计算
    f, t, Zxx = stft(signal, fs=fs, window=window, nperseg=nperseg,
                     noverlap=noverlap, nfft=nfft)
    freq_mask = (f >= fmin) & (f <= fmax)
    f = f[freq_mask]
    Zxx = Zxx[freq_mask, :]

    # 输出STFT的形状
    print(f"STFT结果的形状: {Zxx.shape}")

    # CWT计算
    cwt_result, time_cwt = cwt(signal, scales, fs)
    freqs_cwt = fs / scales

    num_of_pic = ONOFF_1D + ONOFF_STFT + ONOFF_CWT + ONOFF_STFT_1D
    fig, axs = plt.subplots(num_of_pic, 1, figsize=(12, 16))

    plot_idx = 0

    if ONOFF_1D:
        axs[plot_idx].plot(time, signal)
        axs[plot_idx].set_title(f'{filename}的原始信号')
        axs[plot_idx].set_ylabel('振幅')
        axs[plot_idx].set_xlabel('时间 [秒]')
        axs[plot_idx].grid(True)
        plot_idx += 1

    if ONOFF_STFT:
        pcm = axs[plot_idx].pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        axs[plot_idx].set_title(f'{filename}的STFT幅值谱')
        axs[plot_idx].set_ylabel('频率 [Hz]')
        axs[plot_idx].set_xlabel('时间 [秒]')
        axs[plot_idx].set_ylim([fmin, fmax])
        fig.colorbar(pcm, ax=axs[plot_idx], label='幅值')
        plot_idx += 1

    if ONOFF_CWT:
        axs[plot_idx].imshow(np.abs(cwt_result), aspect='auto', extent=[0, time[-1], freqs_cwt[-1], freqs_cwt[0]], cmap='jet')
        axs[plot_idx].set_title(f'{filename}的小波变换幅值谱')
        axs[plot_idx].set_ylabel('频率 [Hz]')
        axs[plot_idx].set_xlabel('时间 [秒]')
        fig.colorbar(axs[plot_idx].images[0], ax=axs[plot_idx], label='幅值')
        plot_idx += 1

    if ONOFF_STFT_1D:
        signal_norm = (signal - signal.min()) / (signal.max() - signal.min()) * fmax
        axs[plot_idx].plot(time, signal_norm, label="1D Signal", color='white', alpha=0.6)
        axs[plot_idx].pcolormesh(t, f, np.abs(Zxx), shading='gouraud', alpha=0.9)
        axs[plot_idx].set_title(f'{filename}的STFT频谱图与原始信号对比(原始信号已经标准化)')
        axs[plot_idx].set_ylabel('频率 [Hz]')
        axs[plot_idx].set_xlabel('时间 [秒]')
        axs[plot_idx].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

# 主程序逻辑
def main(data_path, fs=500, window='hann', nperseg=64, noverlap=None, nfft=512, fmin=0, fmax=100, SCALE_NUM=128):
    """
    主程序入口
    :param data_path: 数据文件路径
    :param fs: 采样频率
    :param window: STFT窗函数类型
    :param nperseg: STFT每段的长度
    :param noverlap: STFT重叠长度
    :param nfft: FFT点数
    :param fmin: 最低频率
    :param fmax: 最高频率
    :param SCALE_NUM: 小波尺度数
    """
    if noverlap is None:
        noverlap = nperseg // 2

    data = read_data(data_path)
    scales = np.arange(1, SCALE_NUM)
    for filename, accresult in data:
        signal = -accresult[1, :]  # 选择第二条数据并反转
        time = np.arange(len(signal)) / fs
        process_and_plot_signal(signal, time, fs, scales, filename,
                                window=window, nperseg=nperseg, noverlap=noverlap, 
                                nfft=nfft, fmin=fmin, fmax=fmax)

# 执行主程序
data_path = r'collect_data\device3\\p4-test\sjx\scg'
main(data_path)
