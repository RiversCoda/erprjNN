import os
import scipy.io as sio
from glob import glob

# 路径中使用原始字符串以避免转义字符冲突
path = r"train_data\device3\p4-test\LJH\scg\*.mat"

# 获取所有符合路径的 .mat 文件
files = glob(path)

for file_path in files:
    # 加载 .mat 文件
    mat_data = sio.loadmat(file_path)
    
    # 检查是否有 scg_data 属性且没有 accresult 属性
    if 'scg_data' in mat_data and 'accresult' not in mat_data:
        # 重命名 scg_data 属性为 accresult
        mat_data['accresult'] = mat_data.pop('scg_data')
        
        # 保存修改后的 .mat 文件，覆盖原文件
        sio.savemat(file_path, mat_data)
        print(f"Updated '{file_path}': 'scg_data' renamed to 'accresult'.")
    elif 'accresult' in mat_data:
        print(f"No changes needed for '{file_path}': 'accresult' already exists.")
    else:
        print(f"'{file_path}' has neither 'scg_data' nor 'accresult'.")

print("Processing complete.")
