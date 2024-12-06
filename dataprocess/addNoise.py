import json
import os
import glob
import scipy.io
import numpy as np
from datetime import datetime
import shutil
import random

def extract_name(file_path):
    # 规范化路径分隔符
    file_path = os.path.normpath(file_path)
    parts = file_path.split(os.sep)
    if 'p4-test' in parts:
        idx = parts.index('p4-test')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return ''

# 加载配置文件
config_file = 'dataprocess\\addNoiseText.JSON'
with open(config_file, 'r') as f:
    config = json.load(f)

quiet_data_paths = config['quiet_data_paths']
noise_data_paths = config['noise_data_paths']
window_size = config['window_size']
step_size = config['step_size']
noise_multiplier = config['noise_multiplier']

# 输出目录设置
today = datetime.now().strftime('%Y%m%d')
output_dir = os.path.join('addNoise_data', today)
noise_output_dir = os.path.join(output_dir, 'noise')
raw_output_dir = os.path.join(output_dir, 'raw')

# 确保输出目录存在
os.makedirs(noise_output_dir, exist_ok=True)
os.makedirs(raw_output_dir, exist_ok=True)

# 备份配置文件到输出目录，并添加时间戳
config_backup_file = os.path.join(output_dir, 'addNoiseTextCopy.JSON')
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
with open(config_file, 'r') as f:
    config_content = json.load(f)
config_content['timestamp'] = timestamp
with open(config_backup_file, 'w', encoding='utf-8') as f:
    json.dump(config_content, f, indent=4, ensure_ascii=False)

# 收集所有安静数据文件
quiet_files = []
for path in quiet_data_paths:
    files = glob.glob(path, recursive=True)
    quiet_files.extend(files)

# 收集所有噪音数据文件
noise_files = []
for path in noise_data_paths:
    files = glob.glob(path, recursive=True)
    noise_files.extend(files)

# 确保存在噪音文件
if not noise_files:
    raise ValueError("未找到任何噪音数据文件。")

# 预加载所有噪音数据到内存中以优化性能
preloaded_noise = []
for noise_file in noise_files:
    noise_data = scipy.io.loadmat(noise_file)
    if 'accresult' in noise_data:
        accresult = noise_data['accresult']
        if accresult.shape[1] >= window_size:
            preloaded_noise.append((noise_file, accresult))
# 如果没有有效的噪音数据，抛出错误
if not preloaded_noise:
    raise ValueError("没有有效的噪音数据文件。")

# 处理安静数据文件
for quiet_file in quiet_files:
    # 加载安静数据
    data = scipy.io.loadmat(quiet_file)
    if 'accresult' not in data:
        continue  # 如果文件中没有'accresult'变量，则跳过
    accresult = data['accresult']  # 形状为 (4, 5000)
    total_length = accresult.shape[1]
    # 滑动窗口
    num_slices = (total_length - window_size) // step_size + 1
    if num_slices <= 0:
        continue  # 如果数据太短，跳过
    # 从安静文件路径中提取名称
    name = extract_name(quiet_file)
    if not name:
        continue  # 如果无法确定名称，跳过
    for i in range(num_slices):
        start = i * step_size
        end = start + window_size
        slice_data = accresult[:, start:end]  # 形状为 (4, window_size)
        # 获取基础文件名（不带扩展名）
        base_name = os.path.basename(quiet_file)
        base_name_no_ext = os.path.splitext(base_name)[0]
        # 安静数据的输出子目录
        raw_output_subdir = os.path.join(raw_output_dir, name, 'scg')
        os.makedirs(raw_output_subdir, exist_ok=True)
        # 噪音数据的输出子目录
        noisy_output_subdir = os.path.join(noise_output_dir, name, 'scg')
        os.makedirs(noisy_output_subdir, exist_ok=True)
        
        for n in range(noise_multiplier):
            # 保存安静数据的副本，命名为_copy{n}.mat
            copy_slice_name = f"{base_name_no_ext}_slice{i}_copy{n}.mat"
            copy_output_file = os.path.join(raw_output_subdir, copy_slice_name)
            scipy.io.savemat(copy_output_file, {'accresult': slice_data})
            
            # 从预加载的噪音数据中随机选择一个
            noise_file, noise_accresult = random.choice(preloaded_noise)
            # 随机选择一个起始点
            noise_start_idx = random.randint(0, noise_accresult.shape[1] - window_size)
            noise_slice = noise_accresult[:, noise_start_idx:noise_start_idx + window_size]
            # 添加噪音到安静数据的副本
            noisy_slice = slice_data + noise_slice
            # 保存噪音数据，命名为_copy{n}_noise.mat
            noisy_slice_name = f"{base_name_no_ext}_slice{i}_copy{n}_noise.mat"
            noisy_output_file = os.path.join(noisy_output_subdir, noisy_slice_name)
            scipy.io.savemat(noisy_output_file, {
                'accresult': noisy_slice,
                'noise_source': noise_file
            })
