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
