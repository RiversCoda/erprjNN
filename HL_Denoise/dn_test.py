# dn_test.py

import os
import argparse
import torch
import torch.nn as nn
from dn_dataloader import get_test_dataloader
from dn_model import BaselineTransformer, RowAttentionTransformer, ColumnAttentionTransformer, CombinationAttentionTransformer, MultiCombinationTransformer
from dn_utils import save_params_log
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Denoising Neural Network Testing")
    parser.add_argument('--test_clean_dir', type=str, default='addNoise_data\\test1\\raw', help='Path to clean test data')
    parser.add_argument('--test_noise_dir', type=str, default='addNoise_data\\test1\\noise', help='Path to noise test data')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--normalize_method', type=str, default='minmax', choices=['minmax', 'standard'], help='Normalization method')
    parser.add_argument('--mean', type=float, default=0.0, help='Mean for standard normalization')
    parser.add_argument('--std', type=float, default=1.0, help='Std for standard normalization')
    parser.add_argument('--apply_filter', action='store_true', help='Whether to apply lowpass filter')
    parser.add_argument('--model_type', type=str, required=True, choices=['baseline', 'row_attention', 'column_attention', 'combination_attention', 'multi_combination'], help='Type of model to test')
    parser.add_argument('--stft_mode', type=str, default='magnitude', choices=['magnitude', 'magnitude_phase', 'real_imag'], help='STFT mode for attention transformers')
    parser.add_argument('--num_visualizations', type=int, default=100, help='Number of samples to visualize')
    parser.add_argument('--visdiv', type=int, default=50, help='Number of samples to visualize')
    args = parser.parse_args()
    return args

def get_model(args):
    if args.model_type == 'baseline':
        model = BaselineTransformer()
    elif args.model_type == 'row_attention':
        model = RowAttentionTransformer(stft_mode=args.stft_mode)
    elif args.model_type == 'column_attention':
        model = ColumnAttentionTransformer(stft_mode=args.stft_mode)
    elif args.model_type == 'combination_attention':
        model = CombinationAttentionTransformer(stft_mode=args.stft_mode)
    elif args.model_type == 'multi_combination':
        model = MultiCombinationTransformer(stft_mode=args.stft_mode)
    else:
        raise ValueError("Unsupported model type.")
    return model

def plot_signals(input_signal, label_signal, output_signal, sample_idx, save_dir):
    """
    绘制输入、标签和输出信号并保存图像。
    """
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(input_signal, color='red')
    plt.title('Input (Noisy)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    plt.subplot(1, 3, 2)
    plt.plot(label_signal, color='green')
    plt.title('Label (Clean)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    plt.subplot(1, 3, 3)
    plt.plot(output_signal, color='blue')
    plt.title('Output (Denoised)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sample_{sample_idx}.png'))
    plt.close()

def main():
    args = parse_args()

    # 创建保存测试详情和可视化结果的文件夹
    os.makedirs('testDetails', exist_ok=True)
    os.makedirs(os.path.join('testDetails', 'visualizations'), exist_ok=True)

    # 记录测试参数
    current_time = datetime.now().strftime("%m%d%H%M%S")
    params_log_path = os.path.join('testDetails', f'test_{current_time}.txt')
    params = vars(args)
    save_params_log(params, params_log_path)

    # 获取数据加载器
    test_loader, test_size = get_test_dataloader(
        clean_dir=args.test_clean_dir,
        noise_dir=args.test_noise_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        normalize_method=args.normalize_method,
        mean=args.mean,
        std=args.std,
        apply_filter=args.apply_filter
    )

    # 加载模型
    model = get_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 定义损失函数
    criterion = nn.MSELoss()
    total_loss = 0.0

    # 可视化参数
    num_visualizations = args.num_visualizations
    visualized = 0

    vis_i = 0
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            if args.model_type == 'baseline':
                outputs = model(inputs, inputs)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

            
            # 可视化前几个样本
            # if visualized < num_visualizations:
            if vis_i % args.visdiv == 0:
                batch_size_current = inputs.size(0)
                for i in range(batch_size_current):
                    if visualized >= num_visualizations:
                        break
                    input_signal = inputs[i].cpu().numpy()
                    label_signal = targets[i].cpu().numpy()
                    output_signal = outputs[i].cpu().numpy()

                    # 反标准化（如果使用标准化）
                    if args.normalize_method == 'standard':
                        input_signal = input_signal * args.std + args.mean
                        label_signal = label_signal * args.std + args.mean
                        output_signal = output_signal * args.std + args.mean
                    elif args.normalize_method == 'minmax':
                        # 假设数据在 [0,1] 之间，无需反标准化
                        pass

                    # 绘制并保存图像
                    plot_signals(input_signal, label_signal, output_signal, visualized, os.path.join('testDetails', 'visualizations'))
                    visualized += 1

                    # 输出output_signal的均值、方差
                    print(f"output_signal mean: {np.mean(output_signal)}, output_signal std: {np.std(output_signal)}")
            
            vis_i += 1

    avg_loss = total_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.6f}")

    # 记录测试结果
    with open(params_log_path, 'a') as f:
        f.write(f"Average Test Loss: {avg_loss:.6f}\n")

    print(f"可视化结果已保存到 'testDetails/visualizations/' 文件夹中。")

if __name__ == "__main__":
    main()
