# dn_train.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from dn_dataloader import get_dataloader
from dn_model import BaselineTransformer, RowAttentionTransformer, ColumnAttentionTransformer, CombinedAttentionTransformer, MultiCombinationTransformer
from dn_utils import load_mat, save_mat
from tqdm import tqdm
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Network Denoising Training Script')
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'row', 'column', 'combined', 'multi'],
                        help='Model type to use.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers.')
    parser.add_argument('--normalize_method', type=str, default='-1_to_1', choices=['-1_to_1', 'z-score'],
                        help='Normalization method.')
    parser.add_argument('--apply_filter', action='store_true', help='Apply low-pass filter.')
    parser.add_argument('--save_every', type=int, default=5, help='Save model every X epochs.')
    parser.add_argument('--train_noise_path', type=str, default=r'addNoise_data\\20241205\\noise\\*\\scg\\*.mat',
                        help='Path pattern for training noise data.')
    parser.add_argument('--train_raw_path', type=str, default=r'addNoise_data\\20241205\\raw\\*\\scg\\*.mat',
                        help='Path pattern for training raw data.')
    args = parser.parse_args()
    return args

def get_model(model_type):
    if model_type == 'baseline':
        return BaselineTransformer()
    elif model_type == 'row':
        return RowAttentionTransformer()
    elif model_type == 'column':
        return ColumnAttentionTransformer()
    elif model_type == 'combined':
        return CombinedAttentionTransformer()
    elif model_type == 'multi':
        return MultiCombinationTransformer()
    else:
        raise ValueError("Unsupported model type.")

def main():
    args = parse_args()

    # Create necessary directories
    os.makedirs('save_models', exist_ok=True)
    os.makedirs('trainDetails', exist_ok=True)

    # Initialize dataloader
    train_loader = get_dataloader(
        noise_path_pattern=args.train_noise_path,
        raw_path_pattern=args.train_raw_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        normalize_method=args.normalize_method,
        apply_filter=args.apply_filter
    )

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model).to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Record training parameters
    params_time = datetime.now().strftime("%m%d%H%M%S")
    params_file = os.path.join('trainDetails', f'params_{params_time}.txt')
    with open(params_file, 'w') as f:
        f.write(f"Model Type: {args.model}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Number of Workers: {args.num_workers}\n")
        f.write(f"Normalization Method: {args.normalize_method}\n")
        f.write(f"Apply Filter: {args.apply_filter}\n")
        f.write(f"Training Noise Path: {args.train_noise_path}\n")
        f.write(f"Training Raw Path: {args.train_raw_path}\n")
        f.write(f"Number of Training Samples: {len(train_loader.dataset)}\n")
        # Add more parameters if needed

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for noise, raw in progress_bar:
            noise = noise.to(device)
            raw = raw.to(device)

            optimizer.zero_grad()
            if args.model == 'baseline':
                # For baseline, Transformer requires src and tgt
                output = model(noise, noise)
            else:
                output = model(noise)
            loss = criterion(output, raw)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

        avg_loss = epoch_loss / len(train_loader) 
        print(f"Epoch [{epoch}/{args.epochs}] Average Loss: {avg_loss}")

        # Save model
        if epoch % args.save_every == 0:
            model_name = args.model
            model_filename = f"{model_name}_epoch{epoch}_loss{avg_loss:.4f}.pth"
            model_path = os.path.join('save_models', model_filename)
            torch.save(model.state_dict(), model_path)
            print(f"Saved model: {model_path}")

if __name__ == '__main__':
    main()
