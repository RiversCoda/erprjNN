import scipy.io as sio
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# 指定要读取的文件夹路径
folder_path = r"collect_data\device3\noise_phone\sjx\scg"
# folder_path = r"quiet"
 
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
                for i in range(1, 4):
                    acc0 = 0-accresult[i]
                    # 高斯滤波acc0
                    # acc0 = gaussian_filter(acc0, sigma=5)

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
