import os
import shutil

def find_and_move_mat_files(source_root, dest_root):
    """
    查找并移动所有符合'scg111/username/*/scg/*.mat'格式的文件，移动到'scgs/username/scg/*.mat'，
    如果重名则添加后缀。
    
    :param source_root: 原始根路径 'scg111'
    :param dest_root: 目标根路径 'scgs'
    """
    for root, dirs, files in os.walk(source_root):
        # 检查是否符合指定的路径模式 'scg111/username/*/scg/*.mat'
        if 'scg' in root and root.endswith('scg'):
            for file in files:
                if file.endswith('.mat'):
                    # 提取相对路径中的 username
                    relative_path = os.path.relpath(root, source_root)
                    parts = relative_path.split(os.sep)
                    if len(parts) >= 2:
                        username = parts[0]
                        
                        # 构建目标目录和文件路径
                        dest_dir = os.path.join(dest_root, username, 'scg')
                        os.makedirs(dest_dir, exist_ok=True)
                        
                        source_file = os.path.join(root, file)
                        dest_file = os.path.join(dest_dir, file)
                        
                        # 如果目标文件已经存在，添加后缀
                        if os.path.exists(dest_file):
                            base_name, ext = os.path.splitext(file)
                            counter = 1
                            new_dest_file = os.path.join(dest_dir, f"{base_name}_{counter}{ext}")
                            while os.path.exists(new_dest_file):
                                counter += 1
                                new_dest_file = os.path.join(dest_dir, f"{base_name}_{counter}{ext}")
                            dest_file = new_dest_file
                        
                        # 移动文件
                        shutil.move(source_file, dest_file)
                        print(f"Moved {source_file} to {dest_file}")

# 使用示例
source_root = 'scg111'  # 这是原始根目录
dest_root = 'scgs'  # 这是目标根目录

find_and_move_mat_files(source_root, dest_root)
