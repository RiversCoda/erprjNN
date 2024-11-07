import torch
import os
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

# Prepare test data loader
test_data_dirs = ['path_to_test_noise_data']
test_loader = get_dataloader(test_data_dirs, batch_size=1, shuffle=False, noise_type='test_noise')

# Evaluate model
generator.eval()
discriminator.eval()
results = []

with torch.no_grad():
    for data, _ in test_loader:
        data = data.unsqueeze(1).to(device)
        denoised_data = generator(data)
        output = discriminator(denoised_data)
        probability = output.mean().item()
        results.append(probability)

# Save results
with open('gan_results.txt', 'a') as f:
    f.write(f'Probabilities: {results}\n')
