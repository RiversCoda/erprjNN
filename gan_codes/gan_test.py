import torch
import os
from gan_datalder import get_dataloader
from gan_model import UNetGenerator, Discriminator, MultiLayerMultiHeadTransformer
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the latest saved model
model_files = sorted([f for f in os.listdir('gan_models') if 'generator' in f])
latest_model = model_files[-1]
generator = MultiLayerMultiHeadTransformer().to(device)
generator.load_state_dict(torch.load(f'gan_models/{latest_model}'))

discriminator = Discriminator().to(device)
discriminator.load_state_dict(torch.load(f'gan_models/discriminator_epoch_{latest_model.split("_")[-1]}'))

# Prepare test data loaders using the same noise data directories as the training set
noise_data_dirs = {
    'phone': 'collect_data/device3/noise_phone/sjx/scg',
    'head': 'collect_data/device3/noise_head/sjx/scg',
    'chew': 'collect_data/device3/noise_chew/sjx/scg'
}
test_loaders = {}
for noise_type, data_dir in noise_data_dirs.items():
    test_loaders[noise_type] = get_dataloader([data_dir], batch_size=1, shuffle=False, noise_type=noise_type)

# Create directory for test results
if not os.path.exists('gan_test_results'):
    os.makedirs('gan_test_results')

# Evaluate model
generator.eval()
discriminator.eval()
results = {}

with torch.no_grad():
    for noise_type, loader in test_loaders.items():
        results[noise_type] = []
        # Create subdirectory for the noise type
        noise_dir = os.path.join('gan_test_results', noise_type)
        if not os.path.exists(noise_dir):
            os.makedirs(noise_dir)
        for idx, (data, _) in enumerate(loader):
            data = data.unsqueeze(1).to(device)  # Shape: [batch_size, 1, window_size]
            denoised_data = generator(data)
            output = discriminator(denoised_data)
            probability = output.mean().item()
            results[noise_type].append(probability)
            # Visualization code here
            data_np = data.squeeze().cpu().numpy()
            denoised_data_np = denoised_data.squeeze().cpu().numpy()
            plt.figure()
            plt.plot(data_np, label='Noisy Input')
            plt.plot(denoised_data_np, label='Denoised Output')
            plt.legend()
            plt.title(f'Noise Type: {noise_type}, Sample: {idx}')
            plt.savefig(os.path.join(noise_dir, f'sample_{idx}.png'))
            plt.close()

# Save results
with open('gan_results.txt', 'a') as f:
    for noise_type, probs in results.items():
        f.write(f'Noise Type: {noise_type}, Probabilities: {probs}\n')
