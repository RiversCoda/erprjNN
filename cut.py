import os
import numpy as np
import scipy.io as sio

def process_mat_files(source_root, dest_root):

    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.endswith('.mat'):
                file_path = os.path.join(root, file)
                mat_data = sio.loadmat(file_path)
                
                # 检查'accresult'是否在mat文件中
                if 'accresult' in mat_data:
                    accresult = mat_data['accresult']
                    
                    # 确认'accresult'的形状
                    if accresult.shape == (4, 15000):
                        window_size = 5000
                        step_size = 5000
                    else:
                        window_size = 5000
                        step_size = 4900
                    
                    # 裁切数据
                    segments = []
                    for start in range(0, accresult.shape[1] - window_size + 1, step_size):
                        segment = accresult[:, start:start+window_size]
                        segments.append(segment)
                    
                    # 创建目标文件夹
                    relative_path = os.path.relpath(root, source_root)
                    dest_dir = os.path.join(dest_root, relative_path)
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    # 保存裁切后的文件
                    for idx, segment in enumerate(segments):
                        new_filename = os.path.join(dest_dir, f"{os.path.splitext(file)[0]}_cut{idx+1}.mat")
                        sio.savemat(new_filename, {'accresult': segment})
                        print(f"Saved cut file {new_filename}")
                elif 'scg_data' in mat_data:
                    accresult = mat_data['scg_data']
                    
                    # 确认'accresult'的形状
                    if accresult.shape == (4, 15000):
                        window_size = 5000
                        step_size = 5000
                    else:
                        window_size = 5000
                        step_size = 4900
                    
                    # 裁切数据
                    segments = []
                    for start in range(0, accresult.shape[1] - window_size + 1, step_size):
                        segment = accresult[:, start:start+window_size]
                        segments.append(segment)
                    
                    # 创建目标文件夹
                    relative_path = os.path.relpath(root, source_root)
                    dest_dir = os.path.join(dest_root, relative_path)
                    os.makedirs(dest_dir, exist_ok=True)
                    
                    # 保存裁切后的文件
                    for idx, segment in enumerate(segments):
                        new_filename = os.path.join(dest_dir, f"{os.path.splitext(file)[0]}_cut{idx+1}.mat")
                        sio.savemat(new_filename, {'scg_data': segment})
                        print(f"Saved cut file {new_filename}")

# 使用示例
source_root = 'scgs'  # 这是原始根目录
dest_root = 'cut_scgs'  # 这是目标根目录

process_mat_files(source_root, dest_root)
