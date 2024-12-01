对于如下工程，我希望在保留原有功能的基础上，添加一个新功能：使用行列注意力进行训练，具体如下：（请你告诉我需要修改哪些内容）
对输入的1*2000的训练数据，使用stft进行变换（参数如下nperseg=64, nfft=512）保留0-100hz的结果送入行列注意力模型进行训练，输出1*2000的数据。
模型结构为：将stft转化为2D的结果按行拆，编码成token，送入transformer模型，输出编码成1*2000的数据。
然后将stft转化为2D的结果按列拆，编码成token，送入transformer模型，输出编码成1*2000的数据。
将两个结果按通道拼接，送入1*2的通道卷积层，输出1*2000的数据。
```py dn_dataloader.py
import os
import glob
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dn_utils import *

Ncutoff = 70

class DenoiseDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.noise_dir = os.path.join(root_dir, mode, 'noise')
        self.raw_dir = os.path.join(root_dir, mode, 'raw')
        self.noisy_files = glob.glob(os.path.join(self.noise_dir, '*', 'scg', '*.mat'))
        self.raw_files = glob.glob(os.path.join(self.raw_dir, '*', 'scg', '*.mat'))
        self.data_pairs = self._create_data_pairs()

    def _create_data_pairs(self):
        data_pairs = []
        raw_dict = {}
        for raw_file in self.raw_files:
            key = self._get_key(raw_file)
            raw_dict[key] = raw_file

        for noisy_file in self.noisy_files:
            key = self._get_key(noisy_file).rsplit('_noise', 1)[0]
            if key in raw_dict:
                data_pairs.append((noisy_file, raw_dict[key]))
        return data_pairs

    def _get_key(self, filepath):
        return os.path.splitext(os.path.basename(filepath))[0]

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        noisy_file, raw_file = self.data_pairs[idx]
        noisy_mat = sio.loadmat(noisy_file)['accresult'][1]
        raw_mat = sio.loadmat(raw_file)['accresult'][1]
        noisy_data = np.array(noisy_mat, dtype=np.float32)
        raw_data = np.array(raw_mat, dtype=np.float32)

        # 应用 N Hz 低通滤波器
        noisy_data = low_pass_filter(noisy_data, cutoff=Ncutoff, fs=1000, order=5)
        raw_data = low_pass_filter(raw_data, cutoff=Ncutoff, fs=1000, order=5)

        # 应用归一化
        noisy_data = normalize(noisy_data)
        raw_data = normalize(raw_data)

        return noisy_data, raw_data


def get_dataloader(root_dir, mode='train', batch_size=32, num_workers=4):
    dataset = DenoiseDataset(root_dir, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader
```

```py dn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDenoiser(nn.Module):
    def __init__(self, input_dim=2000, num_layers=8, nhead=8, dim_feedforward=512):
        super(TransformerDenoiser, self).__init__()
        self.embedding = nn.Linear(input_dim, dim_feedforward)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(dim_feedforward, input_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, input_dim=2000):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(64 * (input_dim // 8), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        return self.model(x)

class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
```

```py dn_train.py
import torch
import torch.optim as optim
from torch.cuda import amp
from dn_dataloader import get_dataloader
from dn_model import TransformerDenoiser, Discriminator
from dn_utils import *                            
import torch.nn as nn
from tqdm import tqdm

# Hyperparameters
# model_type = 'gan'  # 'gan' or 'transformer'
model_type = 'transformer'
num_epochs = 500
batch_size = 64
learning_rate = 1e-5
save_every = 1  # Save model every 5 epochs
root_dir = 'addNoise_data'
num_workers = 4

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_dataloader(root_dir, '20241120', batch_size, num_workers)

    if model_type == 'transformer':
        model = TransformerDenoiser().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scaler = amp.GradScaler()
        for epoch in range(num_epochs):
            loop = tqdm(dataloader, leave=False)
            for noisy, clean in loop:
                noisy = noisy.to(device)
                clean = clean.to(device)
                optimizer.zero_grad()
                with amp.autocast():
                    output = model(noisy)
                    loss = criterion(output, clean)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
                loop.set_postfix(loss=loss.item())
            if (epoch + 1) % save_every == 0:
                save_model(model, epoch+1, loss.item(), model_type)
    elif model_type == 'gan':
        generator = TransformerDenoiser().to(device)
        discriminator = Discriminator().to(device)
        criterion = nn.BCELoss()
        optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
        optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)
        scaler = amp.GradScaler()
        for epoch in range(num_epochs):
            loop = tqdm(dataloader, leave=False)
            for noisy, clean in loop:
                noisy = noisy.to(device)
                clean = clean.to(device)
                valid = torch.ones(noisy.size(0), 1).to(device)
                fake = torch.zeros(noisy.size(0), 1).to(device)
                # Train Generator
                optimizer_G.zero_grad()
                with amp.autocast():
                    gen_data = generator(noisy)
                    g_loss = criterion(discriminator(gen_data), valid)
                scaler.scale(g_loss).backward()
                scaler.step(optimizer_G)
                scaler.update()
                # Train Discriminator
                optimizer_D.zero_grad()
                with amp.autocast():
                    real_loss = criterion(discriminator(clean), valid)
                    fake_loss = criterion(discriminator(gen_data.detach()), fake)
                    d_loss = (real_loss + fake_loss) / 2
                scaler.scale(d_loss).backward()
                scaler.step(optimizer_D)
                scaler.update()
                loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
                loop.set_postfix(G_loss=g_loss.item(), D_loss=d_loss.item())
            if (epoch + 1) % save_every == 0:
                save_model(generator, epoch+1, g_loss.item(), f'{model_type}_generator')
                save_model(discriminator, epoch+1, d_loss.item(), f'{model_type}_discriminator')

if __name__ == '__main__':
    train()

```

```py dn_test.py
import torch
from dn_dataloader import *
from dn_model import TransformerDenoiser
from dn_utils import *
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def test(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_dataloader('addNoise_data', 'test1', batch_size=1, num_workers=4) ## 路径
    model = TransformerDenoiser().to(device)
    load_model(model, model_path)
    criterion = nn.MSELoss()
    total_loss = 0
    os.makedirs('test_results', exist_ok=True)
    
    with torch.no_grad():
        for idx, (noisy, clean) in enumerate(dataloader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            output = model(noisy)
            loss = criterion(output, clean)
            total_loss += loss.item()
            
            # Convert tensors to numpy arrays for plotting and multiply by -1
            noisy_np1 = -1 * noisy.cpu().numpy().flatten()
            clean_np1 = -1 * clean.cpu().numpy().flatten()
            output_np1 = -1 * output.cpu().numpy().flatten()

            noisy_np = noisy_np1
            clean_np = clean_np1
            output_np = 0.7 * output_np1 + 0.3 * clean_np1
            
            # Plotting
            plt.figure(figsize=(22, 18))  # Adjusted figsize to make plots longer
            
            plt.subplot(3, 1, 1)
            plt.plot(noisy_np)
            plt.title('Noisy Data (Flipped)')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            
            plt.subplot(3, 1, 2)
            plt.plot(clean_np)
            plt.title('Clean Data (Flipped)')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            
            plt.subplot(3, 1, 3)
            plt.plot(output_np)
            plt.title('Denoised Output (Flipped)')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            
            plt.tight_layout()
            
            # Save figure
            plot_filename = f'test_results/sample_{idx}.png'
            plt.savefig(plot_filename)
            plt.close()
            
            print(f'Saved plot for sample {idx} to {plot_filename}')
        
        average_loss = total_loss / len(dataloader)
        print(f'Average Test Loss: {average_loss:.4f}')

if __name__ == '__main__':
    model_path = 'saved_models\\transformer_epoch500_loss0.3166.pth'  # Update with your model path
    test(model_path)

```

```py dn_utils.py
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

def save_model(model, epoch, loss, model_name):
    os.makedirs('saved_models', exist_ok=True)
    model_filename = f'{model_name}_epoch{epoch}_loss{loss:.4f}.pth'
    model_path = os.path.join('saved_models', model_filename)
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f'Model loaded from {model_path}')

def low_pass_filter(data, cutoff=100, fs=1000, order=5):
    """
    对数据应用低通滤波器。

    参数：
    - data: 输入的时间序列数据。
    - cutoff: 截止频率，默认 100Hz。
    - fs: 采样率，根据您的数据设置，默认 1000Hz。
    - order: 滤波器的阶数，默认 5。

    返回：
    - 滤波后的数据。
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y.astype(np.float32)

def normalize(data):
    """
    对数据进行标准化处理，使其均值为 0，标准差为 1。

    参数：
    - data: 输入的时间序列数据。

    返回：
    - 标准化后的数据。
    """
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data.astype(np.float32)

# -1 到 1的标准化
def normalize_1_1(data):
    aa = 1
```

