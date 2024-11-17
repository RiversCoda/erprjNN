import os
import torch
import matplotlib.pyplot as plt
from cyclegan_dataloader import get_test_dataloader  # 确保使用了正确的数据加载器
from cyclegan_model import TransformerGenerator  # 修改这里

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载最新的模型
model_files = sorted([f for f in os.listdir('cyclegan_models') if 'G_B2A' in f])
latest_model = model_files[-1]
G_B2A = TransformerGenerator().to(device)  # 使用新的生成器
G_B2A.load_state_dict(torch.load(f'cyclegan_models/{latest_model}', map_location=device))

# 准备测试数据加载器，使用噪声数据目录
data_dirs_B = [
    'collect_data/device3/noise_phone/sjx/scg',
    'collect_data/device3/noise_head/sjx/scg',
    'collect_data/device3/noise_chew/sjx/scg'
]  # 含噪声的信号

test_loader = get_test_dataloader(data_dirs_B, batch_size=1, shuffle=False)

# 确保测试结果保存目录存在
if not os.path.exists('cyclegan_test_results'):
    os.makedirs('cyclegan_test_results')

# 评估模型
G_B2A.eval()

with torch.no_grad():
    for idx, real_B in enumerate(test_loader):
        real_B = real_B.unsqueeze(1).to(device)  # 形状：[batch_size, 1, window_size]
        fake_A = G_B2A(real_B)

        # 可视化
        input_signal = real_B.squeeze(0).squeeze(0).cpu().numpy()
        output_signal = fake_A.squeeze(0).squeeze(0).cpu().numpy()

        plt.figure(figsize=(12, 8))

        # 子图1：输入噪声信号
        plt.subplot(2, 1, 1)
        plt.plot(input_signal, label='输入（含噪声的信号）')
        plt.title(f'测试样本 {idx+1} - 输入信号')
        plt.legend()

        # 子图2：输出去噪信号
        plt.subplot(2, 1, 2)
        plt.plot(output_signal, label='输出（去噪后的信号）')
        plt.title(f'测试样本 {idx+1} - 输出信号')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'cyclegan_test_results/sample_{idx+1}.png')
        plt.close()
