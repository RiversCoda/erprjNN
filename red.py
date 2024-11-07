# 读取"D:\\code\\epProj\\collect_data\device3\\p4-test\szy\scg"路径的第一个mat
# 输出其属性有哪些
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# 指定要读取的文件夹路径
folder_path = r"collect_data\device3\p4-test\JMJ\scg"

# 遍历文件夹中的所有文件

file_name = os.listdir(folder_path)[0]
# 检查是否为.mat

file_path = os.path.join(folder_path, file_name)
print(f"\nReading file: {file_name}")

# 读取.mat文件
data = sio.loadmat(file_path)

# 输出.mat文件的属性
print("Keys in .mat file:", data.keys())

# 输出accresult的形状
accresult = data['accresult']
print("Shape of 'accresult':", accresult.shape)
