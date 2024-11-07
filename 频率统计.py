import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# 读取 .mat 文件
file_path = 'collect_data/device3/p4-test/sjx/scg/10-2-21-24-12.mat'
data = sio.loadmat(file_path)

# 提取 accresult 中的第二条信号（索引为 1）
accresult = data['accresult']
signal = accresult[1, :]  # 第二条5000长度的信号

# 进行傅里叶变换
N = len(signal)
freqs = np.fft.fftfreq(N, d=1/1000)  # 采样频率假设为1000Hz
fft_values = np.fft.fft(signal)

# 取频率绝对值，筛选0-300Hz区间的频率
magnitude = np.abs(fft_values)
freqs = freqs[:N//2]  # 傅里叶变换对称，只取正频部分
magnitude = magnitude[:N//2]

# 累加0-300Hz的频率强度
frequency_range = (freqs >= 0) & (freqs <= 300)
cumulative_intensity = np.cumsum(magnitude[frequency_range])

# 绘制频率分布图
plt.plot(freqs[frequency_range], cumulative_intensity)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Cumulative Intensity')
plt.title('Cumulative Intensity Distribution from 0 to 300 Hz')
plt.grid(True)
plt.show()
