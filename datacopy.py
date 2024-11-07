import os
import random
import shutil

def ensure_dir_exists(path):
    """确保文件夹存在，不存在则创建"""
    if not os.path.exists(path):
        os.makedirs(path)

def copy_files(src_files, dest_dir):
    """将文件从源路径复制到目标文件夹"""
    ensure_dir_exists(dest_dir)
    for file in src_files:
        shutil.copy(file, dest_dir)

def main():
    # 定义源文件夹和目标文件夹
    source_root = 'collect_data/device3/p5'
    train_root = 'train_data/device3/p5'
    test_root = 'test_data/device3/p5'

    # 遍历所有username文件夹
    for username in os.listdir(source_root):
        user_path = os.path.join(source_root, username, 'scg')
        
        # 如果不是文件夹，跳过
        if not os.path.isdir(user_path):
            continue
        
        # 列出该用户的所有文件
        all_files = [os.path.join(user_path, file) for file in os.listdir(user_path)]
        
        # 确保文件数足够
        if len(all_files) < 100:
            print(f"用户 {username} 的文件数少于100，跳过")
            continue
        
        # 随机打乱文件
        random.shuffle(all_files)
        
        # 选择90个文件用于训练集
        train_files = all_files[:90]
        # 剩下的文件中选择10个用于测试集
        test_files = all_files[90:100]

        # 构建目标路径
        train_user_path = os.path.join(train_root, username, 'scg')
        test_user_path = os.path.join(test_root, username, 'scg')

        # 复制文件到目标路径
        copy_files(train_files, train_user_path)
        copy_files(test_files, test_user_path)
        
        print(f"已为用户 {username} 复制 90 个文件到训练集，10 个文件到测试集。")

if __name__ == '__main__':
    main()
