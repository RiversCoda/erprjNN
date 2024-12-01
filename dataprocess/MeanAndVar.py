import os
import scipy.io as sio
import numpy as np

def standardize_and_save_mat_files(source_directory, target_directory):
    # 如果目标目录不存在，创建目标目录
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # 遍历指定目录下的所有 .mat 文件
    for root, _, files in os.walk(source_directory):
        for file in files:
            if file.endswith(".mat"):
                source_file_path = os.path.join(root, file)
                try:
                    # 加载 .mat 文件
                    mat_data = sio.loadmat(source_file_path)
                    # 检查是否包含 'accresult'
                    if 'accresult' in mat_data:
                        accresult = mat_data['accresult']
                        # 检查 accresult 的形状
                        if accresult.shape == (4, 5000):
                            # 获取索引为1的轴（第二条轴）的数据
                            data = accresult[1, :]
                            # 计算当前数据的均值和方差
                            current_mean = np.mean(data)
                            current_variance = np.var(data)
                            # 标准化数据
                            standardized_data = (data - current_mean) / np.sqrt(current_variance)
                            # 缩放到目标均值和方差
                            target_mean = 0.005
                            target_variance = 0.005
                            transformed_data = standardized_data * np.sqrt(target_variance) + target_mean
                            # 更新到 accresult
                            accresult[1, :] = transformed_data
                            mat_data['accresult'] = accresult
                            # 生成目标文件路径
                            relative_path = os.path.relpath(source_file_path, source_directory)
                            target_file_path = os.path.join(target_directory, relative_path)
                            # 确保目标目录存在
                            os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                            # 保存到目标路径
                            sio.savemat(target_file_path, mat_data)
                            print(f"Processed and saved: {file}")
                        else:
                            print(f"File: {file} - 'accresult' shape is not 4*5000, skipping.")
                    else:
                        print(f"File: {file} - 'accresult' not found, skipping.")
                except Exception as e:
                    print(f"Error processing file {file}: {e}")

# 设置源路径和目标路径
source_directory = r"NoiseColl\\noise_11_12_zHead\wxw\scg"
target_directory = r"NoiseColl\\noise_2\wxw\scg"

# 调用函数处理 .mat 文件
standardize_and_save_mat_files(source_directory, target_directory)
