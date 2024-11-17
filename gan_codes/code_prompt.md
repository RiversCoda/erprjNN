- 如下代码是一个基于gan的心跳信号去噪模型，其中包含了数据加载、模型定义、训练和测试等部分。
- 请协助我生成一个新的项目，基于同样的数据集，实现一个基于cyclegan的心跳信号去噪模型。
- 新的项目会添加在原有的项目路径，请实现一个新的cyclegan_dataloder.py、cyclegan_model.py、cyclegan_train.py和cyclegan_test.py文件。
- 在开始之前，如果缺少任何完成项目必须的信息，请先向我询问。

# gan_datalder.py
```python
import os
import glob
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class HeartbeatDataset(Dataset):
    def __init__(self, data_dirs, noise_type=None, window_size=2000, step_size=200, normalize=True, preprocess=True):
        self.data = []
        self.labels = []
        self.noise_type = noise_type
        self.window_size = window_size
        self.step_size = step_size
        self.normalize = normalize
        self.preprocess = preprocess

        for data_dir in data_dirs:
            mat_files = glob.glob(os.path.join(data_dir, '*.mat'))
            for file_path in mat_files:
                mat = scipy.io.loadmat(file_path)
                # 调试信息
                print(f"Processing file: {file_path}")
                print(f"Keys in mat: {mat.keys()}")
                print(f"Type of mat['accresult']: {type(mat['accresult'])}")
                print(f"Shape of mat['accresult']: {mat['accresult'].shape}")

                # 根据实际形状调整索引
                accresult = mat['accresult']  # 形状为 (4, 5000)
                # 假设第二行是 y 轴加速度数据
                signal = accresult[1, :]  # 提取第 1 行的数据

                # 预处理信号
                if self.preprocess:
                    signal = self._preprocess_signal(signal)

                # 归一化信号
                if self.normalize:
                    signal = (signal - np.mean(signal)) / np.std(signal)

                # 滑动窗口分割
                segments = self._segment_signal(signal)
                self.data.extend(segments)

                # 分配标签
                if self.noise_type:
                    self.labels.extend([self.noise_type] * len(segments))
                else:
                    self.labels.extend(['clean'] * len(segments))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.labels[idx]
        return torch.FloatTensor(signal), label

    def _preprocess_signal(self, signal):
        # 在这里添加您的预处理步骤（例如滤波）
        return signal

    def _segment_signal(self, signal):
        segments = []
        for start in range(0, len(signal) - self.window_size + 1, self.step_size):
            end = start + self.window_size
            segments.append(signal[start:end])
        return segments

def get_dataloader(data_dirs, batch_size=32, shuffle=True, **kwargs):
    dataset = HeartbeatDataset(data_dirs, **kwargs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
```

# gan_train.py
```python
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# from gan_codes.gan_datalder import get_dataloader
# from gan_datalder import *
from gan_datalder_test import *

# from gan_codes.gan_model import UNetGenerator, Discriminator
from gan_model import UNetGenerator, Discriminator, MultiLayerMultiHeadTransformer

# Hyperparameters
batch_size = 16
learning_rate_G = 0.00001
learning_rate_D = 0.000005
num_epochs = 500
save_interval = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create models directory if it doesn't exist
if not os.path.exists('gan_models'):
    os.makedirs('gan_models')

# Data directories
clean_data_dirs = ['test_data/device3/p4-test/sjx/scg']
noise_data_dirs = {
    'phone': 'collect_data/device3/noise_phone/sjx/scg',
    'head': 'collect_data/device3/noise_head/sjx/scg',
    'chew': 'collect_data/device3/noise_chew/sjx/scg'
}

# Prepare data loaders
clean_loader = get_dataloader(clean_data_dirs, batch_size=batch_size, shuffle=True)
noise_loaders = {}
for noise_type, data_dir in noise_data_dirs.items():
    noise_loaders[noise_type] = get_dataloader([data_dir], batch_size=batch_size, shuffle=True, noise_type=noise_type)

# Initialize models
generator = MultiLayerMultiHeadTransformer().to(device)
discriminator = Discriminator().to(device)

# Loss functions
criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_G)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_D)

# Training loop
for epoch in range(num_epochs):
    pbar = tqdm(total=len(clean_loader), desc=f'Epoch {epoch+1}/{num_epochs}')
    for i, (clean_data, _) in enumerate(clean_loader):
        # Get corresponding noise data
        noise_type = list(noise_loaders.keys())[i % len(noise_loaders)]
        noise_loader = noise_loaders[noise_type]
        noise_data_iter = iter(noise_loader)
        try:
            noise_data, _ = next(noise_data_iter)
        except StopIteration:
            noise_data_iter = iter(noise_loader)
            noise_data, _ = next(noise_data_iter)

        clean_data = clean_data.unsqueeze(1).to(device)  # Shape: [batch_size, 1, window_size]
        noise_data = noise_data.unsqueeze(1).to(device)

        # Generate fake data
        fake_data = generator(noise_data)

        # Train Discriminator
        discriminator.zero_grad()
        real_output = discriminator(clean_data)
        fake_output = discriminator(fake_data.detach())
        real_labels = torch.ones_like(real_output).to(device)
        fake_labels = torch.zeros_like(fake_output).to(device)
        loss_D_real = criterion_GAN(real_output, real_labels)
        loss_D_fake = criterion_GAN(fake_output, fake_labels)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        generator.zero_grad()
        fake_output = discriminator(fake_data)
        loss_G_GAN = criterion_GAN(fake_output, real_labels)
        loss_G_L1 = criterion_L1(fake_data, clean_data)
        loss_G = loss_G_GAN + 100 * loss_G_L1
        loss_G.backward()
        optimizer_G.step()

        pbar.update(1)
        pbar.set_postfix({'loss_D': loss_D.item(), 'loss_G': loss_G.item()})
    pbar.close()

    # Save models
    if (epoch + 1) % save_interval == 0:
        torch.save(generator.state_dict(), f'gan_models/generator_epoch_{epoch+1}.pth')
        torch.save(discriminator.state_dict(), f'gan_models/discriminator_epoch_{epoch+1}.pth')
```

# gan_model.py
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):  # max_len can be adjusted
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(1)  # shape (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        seq_len = x.size(0)
        pe = self.pe[:seq_len]  # shape (seq_len, 1, d_model)
        return x + pe  # broadcasting over batch_size

class MultiLayerMultiHeadTransformer(nn.Module):
    def __init__(self, n_layers=8, n_heads=8, d_model=128):
        super(MultiLayerMultiHeadTransformer, self).__init__()
        
        self.input_channels = 1
        self.output_channels = 1
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        
        # Linear layer to map from input_channels to d_model
        self.input_proj = nn.Linear(self.input_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Linear layer to map from d_model to output_channels
        self.output_proj = nn.Linear(d_model, self.output_channels)
        
        # Tanh activation function
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # x shape: (batch_size, input_channels, signal_length)
        batch_size, input_channels, signal_length = x.size()
        
        # Transpose x to (batch_size, signal_length, input_channels)
        x = x.permute(0, 2, 1)
        
        # Map input_channels to d_model
        x = self.input_proj(x)
        # x shape: (batch_size, signal_length, d_model)
        
        # Transpose for transformer: (signal_length, batch_size, d_model)
        x = x.permute(1, 0, 2)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Transpose back to (batch_size, signal_length, d_model)
        x = x.permute(1, 0, 2)
        
        # Map from d_model to output_channels
        x = self.output_proj(x)
        # x shape: (batch_size, signal_length, output_channels)
        
        # Transpose back to (batch_size, output_channels, signal_length)
        x = x.permute(0, 2, 1)
        
        # Apply Tanh activation
        x = self.tanh(x)
        
        return x

class UNetGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UNetGenerator, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2)
        )
        self.enc5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(1024, 256, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(512, 128, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose1d(128, output_channels, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
        

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)
        print(f"e1 shape: {e1.shape}")
        e2 = self.enc2(e1)
        print(f"e2 shape: {e2.shape}")
        e3 = self.enc3(e2)
        print(f"e3 shape: {e3.shape}")
        e4 = self.enc4(e3)
        print(f"e4 shape: {e4.shape}")
        e5 = self.enc5(e4)
        print(f"e5 shape: {e5.shape}")

        # Decoding path with skip connections
        d1 = self.dec1(e5)
        print(f"d1 shape before concat: {d1.shape}")
        print(f"e4 shape: {e4.shape}")
        d1 = torch.cat([d1, e4], dim=1)
        print(f"d1 shape after concat: {d1.shape}")

        d2 = self.dec2(d1)
        print(f"d2 shape before concat: {d2.shape}")
        print(f"e3 shape: {e3.shape}")
        d2 = torch.cat([d2, e3], dim=1)
        print(f"d2 shape after concat: {d2.shape}")

        d3 = self.dec3(d2)
        print(f"d3 shape before concat: {d3.shape}")
        print(f"e2 shape: {e2.shape}")
        d3 = torch.cat([d3, e2], dim=1)
        print(f"d3 shape after concat: {d3.shape}")

        d4 = self.dec4(d3)
        print(f"d4 shape before concat: {d4.shape}")
        print(f"e1 shape: {e1.shape}")
        d4 = torch.cat([d4, e1], dim=1)
        print(f"d4 shape after concat: {d4.shape}")

        out = self.dec5(d4)
        print(f"out shape: {out.shape}")
        return out

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # 通道数减半
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),  # 通道数减半
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),  # 通道数减半
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=4, stride=1, padding=1),  # 减少一层卷积
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
```

# gan_test.py
```python
import torch
import os
import matplotlib.pyplot as plt
from gan_datalder import get_dataloader
from gan_model import UNetGenerator, Discriminator, MultiLayerMultiHeadTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the latest saved model
model_files = sorted([f for f in os.listdir('gan_models') if 'generator' in f])
latest_model = model_files[-1]
generator = MultiLayerMultiHeadTransformer().to(device)
generator.load_state_dict(torch.load(f'gan_models/{latest_model}'))

discriminator = Discriminator().to(device)
discriminator.load_state_dict(torch.load(f'gan_models/discriminator_epoch_{latest_model.split("_")[-1]}'))

# Prepare test data loader using the same noise data directories as the training set
noise_data_dirs = {
    'phone': 'collect_data/device3/noise_phone/sjx/scg',
    'head': 'collect_data/device3/noise_head/sjx/scg',
    'chew': 'collect_data/device3/noise_chew/sjx/scg'
}
test_data_dirs = list(noise_data_dirs.values())
test_loader = get_dataloader(test_data_dirs, batch_size=1, shuffle=False, noise_type='test_noise')

# Ensure the directory for saving test results exists
if not os.path.exists('test_results'):
    os.makedirs('test_results')

# Evaluate model
generator.eval()
discriminator.eval()
results = []

with torch.no_grad():
    for idx, (data, _) in enumerate(test_loader):
        data = data.unsqueeze(1).to(device)  # Shape: [batch_size, 1, window_size]
        denoised_data = generator(data)
        output = discriminator(denoised_data)
        probability = output.mean().item()
        results.append(probability)

        # Visualization
        input_signal = data.squeeze(0).squeeze(0).cpu().numpy()
        denoised_signal = denoised_data.squeeze(0).squeeze(0).cpu().numpy()

        plt.figure(figsize=(12, 8))  # Increase figure size for two subplots

        # First subplot: Input signal
        plt.subplot(2, 1, 1)
        plt.plot(input_signal, label='Input (Noisy Signal)')
        plt.title(f'Test Sample {idx+1} - Input Signal')
        plt.legend()

        # Second subplot: Denoised signal
        plt.subplot(2, 1, 2)
        plt.plot(denoised_signal, label='Denoised Signal')
        plt.title(f'Test Sample {idx+1} - Denoised Signal')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'test_results/sample_{idx+1}.png')
        plt.close()

# Save results
with open('gan_results.txt', 'a') as f:
    f.write(f'Probabilities: {results}\n')
```