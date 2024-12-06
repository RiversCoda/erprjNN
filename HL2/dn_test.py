# dn_test.py
import argparse
import os
import torch
from torch.utils.data import DataLoader
from dn_dataloader import get_dataloader
from dn_model import BaselineTransformer, RowAttentionTransformer, ColumnAttentionTransformer, CombinedAttentionTransformer, MultiCombinationTransformer
from dn_utils import load_mat, save_mat
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn as nn

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Network Denoising Testing Script')
    parser.add_argument('--model', type=str, required=True, choices=['baseline', 'row', 'column', 'combined', 'multi'],
                        help='Model type to use.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers.')
    parser.add_argument('--normalize_method', type=str, default='-1_to_1', choices=['-1_to_1', 'z-score'],
                        help='Normalization method.')
    parser.add_argument('--apply_filter', action='store_true', help='Apply low-pass filter.')
    parser.add_argument('--test_noise_path', type=str, default=r'addNoise_data\test1\noise\*\scg\*.mat',
                        help='Path pattern for testing noise data.')
    parser.add_argument('--test_raw_path', type=str, default=r'addNoise_data\test1\raw\*\scg\*.mat',
                        help='Path pattern for testing raw data.')
    args = parser.parse_args()
    return args

def get_model(model_type):
    if model_type == 'baseline':
        return BaselineTransformer()
    elif model_type == 'row':
        return RowAttentionTransformer(stft_params=None)  # Update with actual stft_params if needed
    elif model_type == 'column':
        return ColumnAttentionTransformer(stft_params=None)  # Update with actual stft_params if needed
    elif model_type == 'combined':
        return CombinedAttentionTransformer(stft_params=None)  # Update with actual stft_params if needed
    elif model_type == 'multi':
        return MultiCombinationTransformer(stft_params=None)  # Update with actual stft_params if needed
    else:
        raise ValueError("Unsupported model type.")

def main():
    args = parse_args()

    # 创建必要的目录
    os.makedirs('testDetails/viz', exist_ok=True)

    # 初始化 DataLoader
    test_loader = get_dataloader(
        noise_path_pattern=args.test_noise_path,
        raw_path_pattern=args.test_raw_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        normalize_method=args.normalize_method,
        apply_filter=args.apply_filter
    )

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 定义损失函数
    criterion = nn.MSELoss()

    # 记录测试参数
    test_time = datetime.now().strftime("%m%d%H%M%S")
    test_file = os.path.join('testDetails', f'test_{test_time}.txt')
    with open(test_file, 'w') as f:
        f.write(f"Model Type: {args.model}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Number of Workers: {args.num_workers}\n")
        f.write(f"Normalization Method: {args.normalize_method}\n")
        f.write(f"Apply Filter: {args.apply_filter}\n")
        f.write(f"Testing Noise Path: {args.test_noise_path}\n")
        f.write(f"Testing Raw Path: {args.test_raw_path}\n")
        f.write(f"Number of Testing Samples: {len(test_loader.dataset)}\n")
        # 添加更多参数信息如果需要

    total_loss = 0.0
    total_samples = 0
    viz_counter = 0  # 用于生成唯一的文件名

    # 测试循环
    with torch.no_grad():
        for batch_idx, (noise, raw) in enumerate(tqdm(test_loader, desc="Testing")):
            noise = noise.to(device)
            raw = raw.to(device)

            # 前向传播
            if args.model == 'baseline':
                # 对于 baseline 模型，Transformer 需要 src 和 tgt
                output = model(noise, noise)
            else:
                output = model(noise)

            # 计算损失
            loss = criterion(output, raw)
            batch_size = noise.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # 保存可视化结果
            noise_np = noise.cpu().numpy()
            raw_np = raw.cpu().numpy()
            output_np = output.cpu().numpy()

            for i in range(batch_size):
                viz_counter += 1
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.plot(noise_np[i, :, 0], color='r')
                plt.title('Noisy Input')
                plt.xlabel('Sample Index')
                plt.ylabel('Amplitude')
                plt.subplot(1, 3, 2)
                plt.plot(raw_np[i, :, 0], color='g')
                plt.title('Ground Truth')
                plt.xlabel('Sample Index')
                plt.ylabel('Amplitude')
                plt.subplot(1, 3, 3)
                plt.plot(output_np[i, :, 0], color='b')
                plt.title('Denoised Output')
                plt.xlabel('Sample Index')
                plt.ylabel('Amplitude')
                plt.tight_layout()
                viz_path = os.path.join('testDetails', 'viz', f'sample_{viz_counter}.png')
                plt.savefig(viz_path)
                plt.close()

    # 计算整体平均损失
    average_loss = total_loss / total_samples
    print(f"Average Test Loss: {average_loss:.6f}")

    # 将平均损失写入测试日志文件
    with open(test_file, 'a') as f:
        f.write(f"Average Test Loss: {average_loss:.6f}\n")

    print(f"Testing completed. Details saved in {test_file}")

if __name__ == '__main__':
    main()
