import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from cyclegan_dataloader import get_dataloader_cyclegan
from cyclegan_model import *

# 超参数设置
batch_size = 2
learning_rate = 0.0002
beta1 = 0.5
num_epochs = 100
lambda_cycle = 10.0
lambda_identity = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型保存目录
if not os.path.exists('cyclegan_models'):
    os.makedirs('cyclegan_models')

# 数据目录
data_dirs_A = ['test_data/device3/p4-test/sjx/scg']  # 干净的信号
data_dirs_B = [k
    1
    'collect_data/device3/noise_phone/sjx/scg',
    'collect_data/device3/noise_head/sjx/scg',
    'collect_data/device3/noise_chew/sjx/scg'
]  # 含噪声的信号

# 准备数据加载器
dataloader = get_dataloader_cyclegan(data_dirs_A, data_dirs_B, batch_size=batch_size, shuffle=True)

# 初始化模型
G_A2B = TransformerGenerator().to(device)  # 使用新的生成器
G_B2A = TransformerGenerator().to(device)  # 使用新的生成器
D_A = NLayerDiscriminator().to(device)
D_B = NLayerDiscriminator().to(device)

# 损失函数
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# 优化器
optimizer_G = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=learning_rate, betas=(beta1, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=learning_rate, betas=(beta1, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=learning_rate, betas=(beta1, 0.999))

# 生成的假样本缓存
fake_A_pool = []
fake_B_pool = []

# 训练循环
for epoch in range(num_epochs):
    pbar = tqdm(total=len(dataloader), desc=f'Epoch {epoch+1}/{num_epochs}')
    for i, (real_A, real_B) in enumerate(dataloader):
        real_A = real_A.unsqueeze(1).to(device)  # 形状：[batch_size, 1, window_size]
        real_B = real_B.unsqueeze(1).to(device)

        # 生成器前向传播
        fake_B = G_A2B(real_A)
        rec_A = G_B2A(fake_B)
        fake_A = G_B2A(real_B)
        rec_B = G_A2B(fake_A)

        # 恒等映射损失
        idt_A = G_B2A(real_A)
        idt_B = G_A2B(real_B)
        loss_idt_A = criterion_identity(idt_A, real_A) * lambda_cycle * lambda_identity
        loss_idt_B = criterion_identity(idt_B, real_B) * lambda_cycle * lambda_identity

        # GAN损失
        pred_fake_B = D_B(fake_B)
        target_real = torch.ones_like(pred_fake_B).to(device)
        loss_GAN_A2B = criterion_GAN(pred_fake_B, target_real)

        pred_fake_A = D_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake_A, target_real)

        # 循环一致性损失
        loss_cycle_A = criterion_cycle(rec_A, real_A) * lambda_cycle
        loss_cycle_B = criterion_cycle(rec_B, real_B) * lambda_cycle

        # 总生成器损失
        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # 更新判别器A
        optimizer_D_A.zero_grad()
        pred_real_A = D_A(real_A)
        loss_D_real_A = criterion_GAN(pred_real_A, target_real)

        fake_A_detached = fake_A.detach()
        if len(fake_A_pool) < 50:
            fake_A_pool.append(fake_A_detached)
        else:
            fake_A_pool[i % 50] = fake_A_detached
        fake_A_sample = fake_A_pool[i % len(fake_A_pool)]

        pred_fake_A = D_A(fake_A_sample)
        target_fake = torch.zeros_like(pred_fake_A).to(device)
        loss_D_fake_A = criterion_GAN(pred_fake_A, target_fake)

        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()

        # 更新判别器B
        optimizer_D_B.zero_grad()
        pred_real_B = D_B(real_B)
        loss_D_real_B = criterion_GAN(pred_real_B, target_real)

        fake_B_detached = fake_B.detach()
        if len(fake_B_pool) < 50:
            fake_B_pool.append(fake_B_detached)
        else:
            fake_B_pool[i % 50] = fake_B_detached
        fake_B_sample = fake_B_pool[i % len(fake_B_pool)]

        pred_fake_B = D_B(fake_B_sample)
        loss_D_fake_B = criterion_GAN(pred_fake_B, target_fake)

        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()

        pbar.update(1)
        pbar.set_postfix({'loss_G': loss_G.item(), 'loss_D_A': loss_D_A.item(), 'loss_D_B': loss_D_B.item()})
    pbar.close()

    # 保存模型
    if (epoch + 1) % 10 == 0:
        torch.save(G_A2B.state_dict(), f'cyclegan_models/G_A2B_epoch_{epoch+1}.pth')
        torch.save(G_B2A.state_dict(), f'cyclegan_models/G_B2A_epoch_{epoch+1}.pth')
        torch.save(D_A.state_dict(), f'cyclegan_models/D_A_epoch_{epoch+1}.pth')
        torch.save(D_B.state_dict(), f'cyclegan_models/D_B_epoch_{epoch+1}.pth')
