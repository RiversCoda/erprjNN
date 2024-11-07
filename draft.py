# scgs\fzj_500\scg
# 读取路径下第一个mat文件并输出其属性
import os
import scipy.io as sio

# Define the directory where the .mat files are located
mat_files_directory = r'collect_data\device3\p4-test\ljh\scg'

# Get all .mat files in the directory
mat_files = [f for f in os.listdir(mat_files_directory) if f.endswith('.mat')]

# Process the first .mat file
if mat_files:
    file_path = os.path.join(mat_files_directory, mat_files[0])
    mat_data = sio.loadmat(file_path)
    print(f"Attributes in {mat_files[0]}:")
    for key in mat_data:
        print(key)

# 读取accresult的形状
if 'scg_data' in mat_data:
    accresult = mat_data['scg_data']
    print(f"Shape of 'accresult': {accresult.shape}")

# 路径符合'scgs\username\scg\*.mat'的文件都具有'accresult'属性
# 'accresult'属性的形状是(4, 15000)
# 请读取全部这些文件，使用窗口大小为5000，步长5000的滑动窗口，裁切这些数据
# 并将裁切后的数据保存为新的.mat文件，文件名为原文件名加上'_cut1'、'_cut2'、'_cut3'后缀
# 移动到'cut_scgs\username\scg'目录下