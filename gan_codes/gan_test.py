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
