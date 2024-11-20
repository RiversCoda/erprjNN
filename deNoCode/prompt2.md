对于如下工程，请协助我修改，在训练和测试中，为代码添加：100Hz低通滤波器和归一化功能。请告诉我要修改哪些代码。
``` py
import os
import glob
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dn_utils import *

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
        return noisy_data, raw_data

def get_dataloader(root_dir, mode='train', batch_size=32, num_workers=4):
    dataset = DenoiseDataset(root_dir, mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return dataloader

```

# dn_model.py
``` py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDenoiser(nn.Module):
    def __init__(self, input_dim=2000, num_layers=8, nhead=4, dim_feedforward=512):
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

# dn_train.py
``` py
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
num_epochs = 100
batch_size = 64
learning_rate = 5e-5
save_every = 5  # Save model every 5 epochs
root_dir = 'addNoise_data'
num_workers = 4

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = get_dataloader(root_dir, '20241113', batch_size, num_workers)

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

```py
import torch
from dn_dl import *
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
            
            # Convert tensors to numpy arrays for plotting
            noisy_np = noisy.cpu().numpy().flatten()
            clean_np = clean.cpu().numpy().flatten()
            output_np = output.cpu().numpy().flatten()
            
            # Plotting
            plt.figure(figsize=(8, 12))
            
            plt.subplot(3, 1, 1)
            plt.plot(noisy_np)
            plt.title('Noisy Data')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            
            plt.subplot(3, 1, 2)
            plt.plot(clean_np)
            plt.title('Clean Data')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            
            plt.subplot(3, 1, 3)
            plt.plot(output_np)
            plt.title('Denoised Output')
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
    model_path = 'saved_models\\transformer_epoch10_loss0.0000.pth'  # Update with your model path
    test(model_path)

```

# dn_utils.py
``` py
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

```

