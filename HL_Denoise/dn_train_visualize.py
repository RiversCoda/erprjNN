# dn_train_visualize.py

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from dn_dataloader import get_dataloader
from dn_model import *
from dn_utils import save_params_log
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Denoising Neural Network Training with Visualization")
    parser.add_argument('--train_clean_dir', type=str, default='addNoise_data\\20241201\\raw', help='Path to clean training data')
    parser.add_argument('--train_noise_dir', type=str, default='addNoise_data\\20241201\\noise', help='Path to noise training data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--normalize_method', type=str, default='minmax_minus1_1', choices=['minmax', 'standard', 'minmax_minus1_1'], help='Normalization method')
    parser.add_argument('--mean', type=float, default=0.0, help='Mean for standard normalization')
    parser.add_argument('--std', type=float, default=1.0, help='Std for standard normalization')
    parser.add_argument('--apply_filter', action='store_true', help='Whether to apply lowpass filter')
    parser.add_argument('--model_type', type=str, default='baseline', choices=['baseline', 'simple', 'row_attention', 'column_attention', 'combination_attention', 'multi_combination'], help='Type of model to train')
    parser.add_argument('--nhead', type=int, default=4, help='Number of heads in Transformer')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of Transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Dimension of feedforward network in Transformer')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')  # 设置默认epochs为1
    parser.add_argument('--save_every', type=int, default=1, help='Save model every N epochs')
    parser.add_argument('--stft_mode', type=str, default='magnitude', choices=['magnitude', 'magnitude_phase', 'real_imag'], help='STFT mode for attention transformers')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--visualize_every', type=int, default=1, help='Visualize every N batches')  # 新增参数，控制可视化频率
    args = parser.parse_args()
    return args

def get_model(args):
    if args.model_type == 'baseline':
        model = BaselineTransformer(nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.dim_feedforward)
    elif args.model_type == 'row_attention':
        model = RowAttentionTransformer(stft_mode=args.stft_mode, nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.dim_feedforward)
    elif args.model_type == 'simple':
        model = SimpleDenoisingModel(input_dim=2000, output_dim=2000)
    elif args.model_type == 'column_attention':
        model = ColumnAttentionTransformer(stft_mode=args.stft_mode, nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.dim_feedforward)
    elif args.model_type == 'combination_attention':
        model = CombinationAttentionTransformer(stft_mode=args.stft_mode, nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.dim_feedforward)
    elif args.model_type == 'multi_combination':
        model = MultiCombinationTransformer(stft_mode=args.stft_mode, nhead=args.nhead, num_layers=args.num_layers, dim_feedforward=args.dim_feedforward)
    else:
        raise ValueError("Unsupported model type.")
    return model

def reverse_normalize(data, method, mean=0.0, std=1.0):
    if method == 'standard':
        return data * std + mean
    elif method == 'minmax':
        # 假设minmax是 [0,1]，反标准化需要知道原始min和max，此处简化处理
        return data  # 如果有原始min和max，应在此处反标准化
    elif method == 'minmax_minus1_1':
        return (data * 2) - 1  # 假设minmax_minus1_1是将数据缩放到 [-1,1]
    else:
        raise ValueError("Unsupported normalization method for reverse.")

def plot_signals(input_signal, label_signal, output_signal, epoch, batch_idx, sample_idx, save_dir):
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
    # 保存路径包含epoch, batch, sample索引
    filename = f'epoch{epoch}_batch{batch_idx}_sample{sample_idx}.png'
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

def main():
    args = parse_args()

    # 创建保存模型和可视化结果的文件夹
    os.makedirs('save_models', exist_ok=True)
    os.makedirs('trainDetails', exist_ok=True)
    visualizations_dir = os.path.join('trainDetails', 'visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)

    # 记录训练参数
    current_time = datetime.now().strftime("%m%d%H%M%S")
    params_log_path = os.path.join('trainDetails', f'params_{current_time}.txt')
    params = vars(args)
    save_params_log(params, params_log_path)

    # 获取数据加载器
    train_loader, train_size = get_dataloader(
        clean_dir=args.train_clean_dir,
        noise_dir=args.train_noise_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        normalize_method=args.normalize_method,
        mean=args.mean,
        std=args.std,
        apply_filter=args.apply_filter
    )

    # 获取模型
    model = get_model(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 混合精度
    scaler = torch.cuda.amp.GradScaler()

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (inputs, targets) in enumerate(progress_bar, 1):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                if args.model_type == 'baseline':
                    outputs = model(inputs, inputs)
                elif args.model_type == 'simple': ##
                    outputs = model(inputs)
                else:
                    outputs = model(inputs)
            loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

            # 可视化
            # 根据用户需求，输出全部训练次数的可视化结果
            # 为避免生成过多图像，可以选择每个批次的前几个样本
            # 如果确实需要每个样本的图像，可以移除下面的限制
            visualize_every = args.visualize_every
            if batch_idx % visualize_every == 0:
                # 选择每个批次的前几个样本进行可视化
                num_samples_to_visualize = min(5, inputs.size(0))  # 可视化最多5个样本
                for sample_idx in range(num_samples_to_visualize):
                    input_signal = inputs[sample_idx].cpu().numpy()
                    label_signal = targets[sample_idx].cpu().numpy()
                    output_signal = outputs[sample_idx].cpu().detach().numpy()

                    # 反标准化
                    input_signal = reverse_normalize(input_signal, args.normalize_method, args.mean, args.std)
                    label_signal = reverse_normalize(label_signal, args.normalize_method, args.mean, args.std)
                    output_signal = reverse_normalize(output_signal, args.normalize_method, args.mean, args.std)

                    # 绘制并保存图像
                    plot_signals(input_signal, label_signal, output_signal, epoch, batch_idx, sample_idx, visualizations_dir)

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch}/{args.epochs}], Loss: {avg_loss:.6f}")

        # 保存模型
        if epoch % args.save_every == 0:
            model_name = args.model_type
            # 将学习率格式化为科学计数法，以便文件名中可读
            lr_formatted = "{:.0e}".format(args.learning_rate)
            model_filename = f"{model_name}_lr{lr_formatted}_nhead{args.nhead}_layers{args.num_layers}_epoch{epoch}_loss{avg_loss:.6f}.pt"
            save_path = os.path.join('save_models', model_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"Saved model to {save_path}")

if __name__ == "__main__":
    main()
