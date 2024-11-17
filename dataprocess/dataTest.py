import os
import glob
from scipy.io import loadmat

def check_mat_files(base_paths):
    for base_path in base_paths:
        # 使用glob模式匹配所有符合条件的.mat文件路径
        mat_files = glob.glob(os.path.join(base_path, "*", "scg", "*.mat"))
        
        for file_path in mat_files:
            try:
                data = loadmat(file_path)  # 加载.mat文件内容
                # 检查是否存在accresult属性
                if 'accresult' not in data:
                    print(f"文件 {file_path} 缺少 'accresult' 属性")

                    # 输出该文件实际拥有的属性
                    print(f"实际属性: {data.keys()}")
                    continue
                
                # 检查accresult的形状是否为(4, 5000)
                if data['accresult'].shape != (4, 5000):
                    print(f"文件 {file_path} 的 'accresult' 形状不正确，实际形状为 {data['accresult'].shape}")

                # else:
                    # print(f"文件 {file_path} 通过检查")
            except Exception as e:
                print(f"无法读取文件 {file_path}，错误信息: {e}")

if __name__ == "__main__":
    # 指定待检测的路径
    base_paths = [
        r'train_data\device3\p4-test',
        r'test_data\device3\p4-test'
    ]
    check_mat_files(base_paths)
