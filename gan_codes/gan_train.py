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
