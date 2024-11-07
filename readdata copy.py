import scipy.io as sio
import os
import matplotlib.pyplot as plt
import numpy as np

# Sine wave generation function
def generate_weighted_sine_wave(length, amplitude, period):
    x = np.linspace(0, 2 * np.pi * (length / period), length)
    sine_wave = amplitude * np.sin(x)
    weights = np.linspace(1, 0, length)  # Linearly weight from 1 to 0
    weighted_sine_wave = sine_wave * weights
    return weighted_sine_wave

# 指定要读取的文件夹路径
folder_path = r"D:\\code\\epProj\\collect_data\device3\\p4-test\\lr\scg"

# 遍历文件夹中的所有文件
for file_name in os.listdir(folder_path):
    # 检查是否为.mat文件
    if file_name.endswith(".mat"):
        file_path = os.path.join(folder_path, file_name)
        print(f"\nReading file: {file_name}")
        
        # 读取.mat文件
        data = sio.loadmat(file_path)
        
        # 输出.mat文件的属性
        print("Keys in .mat file:", data.keys())
        
        # 如果存在"accresult"键，则输出它的形状并绘制3条数据 (line 2-4)
        if "accresult" in data:
            accresult = data["accresult"]
            print("Shape of 'accresult':", accresult.shape)
            
            # 检查 accresult 的形状是否是 (4, N)
            if accresult.shape[0] == 4 and 1000 < accresult.shape[1] < 500000:
                # 创建一个新的图形窗口
                fig, axs = plt.subplots(3, 1, figsize=(10, 8))
                
                # 绘制 line 2-4
                for i in range(1, 2):
                    acc0 = 0 - accresult[i]

                    # Define parameters for sine wave and step size
                    step_size = 400
                    sine_length = 300
                    sine_period = 45
                    
                    # Detect max values with step size of 400
                    for j in range(0, len(acc0) - step_size, step_size):
                        window = acc0[j:j + step_size]
                        max_index = np.argmax(window)
                        peak_index = j + max_index
                        
                        if peak_index + sine_length < len(acc0):
                            # Get the peak value (used as half the amplitude of the sine wave)
                            peak_value = acc0[peak_index]
                            amplitude = peak_value / 2
                            
                            # Apply sine wave to the next 300 points
                            sine_wave = generate_weighted_sine_wave(sine_length, amplitude, sine_period)
                            acc0[peak_index + 1:peak_index + 1 + sine_length] += sine_wave
                            
                            # Scale the 300-400 points after the sine wave by 0.6
                            acc0[peak_index + 1 + sine_length:peak_index + step_size] *= 0.6
                    
                    # 对原始信号乘以自身的绝对值的0.5次方
                    acc0 = acc0 * np.abs(acc0) ** 0.3

                    # Plot the modified signal
                    axs[i-1].plot(acc0)
                    axs[i-1].set_title(f'Line {i+1} of {file_name}')
                    axs[i-1].set_xlabel('Sample Index')
                    axs[i-1].set_ylabel('Amplitude')
                
                # 调整布局
                plt.tight_layout()
                plt.show()
            else:
                print(f"'accresult' does not have the expected shape (4, N) in file {file_name}")
        else:
            print(f"'accresult' not found in the file {file_name}")
