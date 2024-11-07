import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
from dataloader import get_dataloaders
from models import HeartbeatResNet
from datetime import datetime

# 超参数
batch_size = 128
learning_rate = 1e-4
num_epochs = 40
save_every = 5  # 每5个epoch保存一次模型
train_dir = 'train_data/device3/p4-test/'
save_dir = 'save_models'
num_workers = 4  # 多线程数量，可根据硬件调整
pin_memory = True  # GPU数据加载优化

os.makedirs(save_dir, exist_ok=True)

if __name__ == "__main__":
    # 准备数据加载器
    train_loader, val_loader = get_dataloaders(train_dir, batch_size, num_workers=num_workers, pin_memory=pin_memory)

    # 初始化模型、损失函数、优化器和梯度缩放器
    model = HeartbeatResNet().cuda()
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    for epoch in range(1, num_epochs + 1):
        model.train()
        loop = tqdm(train_loader, leave=False)
        for anchor, positive, negative in loop:
            anchor = anchor.cuda(non_blocking=True)  # 使用 non_blocking=True 加速数据传输
            positive = positive.cuda(non_blocking=True)
            negative = negative.cuda(non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                anchor_out = model(anchor)
                positive_out = model(positive)
                negative_out = model(negative)
                loss = criterion(anchor_out, positive_out, negative_out)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(loss=loss.item())

        # 验证
        # model.eval()
        # val_loss = 0
        # with torch.no_grad():
        #     for anchor, positive, negative in val_loader:
        #         anchor = anchor.cuda(non_blocking=True)
        #         positive = positive.cuda(non_blocking=True)
        #         negative = negative.cuda(non_blocking=True)
        #         with autocast():
        #             anchor_out = model(anchor)
        #             positive_out = model(positive)
        #             negative_out = model(negative)
        #             loss = criterion(anchor_out, positive_out, negative_out)
        #         val_loss += loss.item()
        # avg_val_loss = val_loss / len(val_loader)
        # print(f'Epoch [{epoch}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

        # 保存模型
        if epoch % save_every == 0:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'model_epoch_{epoch}_time_{current_time}_lr_{learning_rate}_bs_{batch_size}.pth'
            torch.save(model.state_dict(), os.path.join(save_dir, filename))
