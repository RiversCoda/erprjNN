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
