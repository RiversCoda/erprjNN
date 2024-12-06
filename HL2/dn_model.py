# dn_model.py
import torch
import torch.nn as nn
from torch.nn import Transformer
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class BaselineTransformer(nn.Module):
    def __init__(self, input_dim=1, model_dim=256, num_heads=4, num_layers=4, output_dim=1):
        super(BaselineTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim)
        self.transformer = Transformer(d_model=model_dim, nhead=num_heads, num_encoder_layers=num_layers, 
                               num_decoder_layers=num_layers, batch_first=True)

        self.output_linear = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt):
        # src and tgt shape: (batch, 1, 2000)
        src = self.input_linear(src)  # (batch, 1, model_dim)
        tgt = self.input_linear(tgt)  # (batch, 1, model_dim)
        src = src.permute(1, 0, 2)    # (1, batch, model_dim)
        tgt = tgt.permute(1, 0, 2)    # (1, batch, model_dim)

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        output = self.transformer(src, tgt)  # (1, batch, model_dim)
        output = output.permute(1, 0, 2)      # (batch, 1, model_dim)
        output = self.output_linear(output)   # (batch, 1, output_dim)
        return output

class AttentionTransformer(nn.Module):
    def __init__(self, model_dim=256, num_heads=4, num_layers=4, output_dim=2000):
        super(AttentionTransformer, self).__init__()
        self.model_dim = model_dim
        self.transformer = Transformer(d_model=model_dim, nhead=num_heads, 
                                       num_encoder_layers=num_layers, 
                                       num_decoder_layers=num_layers)
        self.pos_encoder = PositionalEncoding(model_dim)
        self.output_linear = nn.Linear(model_dim, output_dim)

    def forward(self, tokens):
        # tokens shape: (seq_len, batch, model_dim)
        tokens = self.pos_encoder(tokens)
        output = self.transformer(tokens, tokens)  # (seq_len, batch, model_dim)
        output = output.permute(1, 0, 2)          # (batch, seq_len, model_dim)
        output = self.output_linear(output)       # (batch, seq_len, output_dim)
        return output

class RowAttentionTransformer(nn.Module):
    def __init__(self, stft_params, model_dim=256, num_heads=4, num_layers=4, output_dim=2000):
        super(RowAttentionTransformer, self).__init__()
        self.stft_params = stft_params
        self.model_dim = model_dim
        self.transformer = AttentionTransformer(model_dim, num_heads, num_layers, output_dim)
        self.final_linear = nn.Linear(output_dim, 1)

    def forward(self, x):
        # x shape: (batch, 1, 2000)
        # Apply STFT
        # Here, you should apply STFT and process the spectrogram
        # For simplicity, let's assume x is already transformed to spectrogram
        # and we keep only row-wise tokens (frequency bins)
        # Placeholder implementation
        spectrogram = x  # Replace with actual STFT
        # spectrogram shape: (batch, freq_bins, time_steps)
        spectrogram = spectrogram.squeeze(1)  # (batch, freq_bins, time_steps)
        batch, freq_bins, time_steps = spectrogram.size()
        tokens = spectrogram.view(batch, freq_bins, -1)  # (batch, freq_bins, features)
        tokens = tokens.permute(1, 0, 2)  # (freq_bins, batch, features)
        output = self.transformer(tokens)  # (freq_bins, batch, output_dim)
        output = output.permute(1, 0, 2).contiguous()  # (batch, freq_bins, output_dim)
        output = output.view(batch, -1)  # (batch, freq_bins * output_dim)
        output = self.final_linear(output)  # (batch, 1)
        return output

class ColumnAttentionTransformer(nn.Module):
    def __init__(self, stft_params, model_dim=256, num_heads=4, num_layers=4, output_dim=2000):
        super(ColumnAttentionTransformer, self).__init__()
        self.stft_params = stft_params
        self.model_dim = model_dim
        self.transformer = AttentionTransformer(model_dim, num_heads, num_layers, output_dim)
        self.final_linear = nn.Linear(output_dim, 1)

    def forward(self, x):
        # Similar to RowAttentionTransformer but along columns (time steps)
        spectrogram = x  # Replace with actual STFT
        spectrogram = spectrogram.squeeze(1)  # (batch, freq_bins, time_steps)
        batch, freq_bins, time_steps = spectrogram.size()
        tokens = spectrogram.view(batch, time_steps, -1)  # (batch, time_steps, features)
        tokens = tokens.permute(1, 0, 2)  # (time_steps, batch, features)
        output = self.transformer(tokens)  # (time_steps, batch, output_dim)
        output = output.permute(1, 0, 2).contiguous()  # (batch, time_steps, output_dim)
        output = self.final_linear(output)  # (batch, time_steps, 1)
        output = output.view(batch, 1, time_steps)  # (batch, 1, 2000)
        return output

class CombinedAttentionTransformer(nn.Module):
    def __init__(self, stft_params, model_dim=256, num_heads=4, num_layers=4, output_dim=2000):
        super(CombinedAttentionTransformer, self).__init__()
        self.row_transformer = RowAttentionTransformer(stft_params, model_dim, num_heads, num_layers, output_dim)
        self.column_transformer = ColumnAttentionTransformer(stft_params, model_dim, num_heads, num_layers, output_dim)
        self.final_conv = nn.Conv1d(2, 1, kernel_size=1)

    def forward(self, x):
        row_out = self.row_transformer(x)      # (batch, 1, 2000)
        col_out = self.column_transformer(x)  # (batch, 1, 2000)
        combined = torch.cat((row_out, col_out), dim=1)  # (batch, 2, 2000)
        out = self.final_conv(combined)    # (batch, 1, 2000)
        return out

class MultiCombinationTransformer(nn.Module):
    def __init__(self, stft_params, model_dim=256, num_heads=4, num_layers=4, output_dim=2000):
        super(MultiCombinationTransformer, self).__init__()
        self.baseline = BaselineTransformer(input_dim=1, model_dim=model_dim, num_heads=num_heads, 
                                           num_layers=num_layers, output_dim=model_dim)
        self.row_transformer = RowAttentionTransformer(stft_params, model_dim, num_heads, num_layers, output_dim)
        self.column_transformer = ColumnAttentionTransformer(stft_params, model_dim, num_heads, num_layers, output_dim)
        self.final_conv = nn.Conv1d(3, 1, kernel_size=1)

    def forward(self, x):
        baseline_out = self.baseline(x, x)      # (batch, 1, model_dim)
        row_out = self.row_transformer(x)       # (batch, 1, 2000)
        col_out = self.column_transformer(x)    # (batch, 1, 2000)
        # To combine, ensure all outputs have the same dimension
        # For simplicity, let's assume model_dim equals output_dim
        combined = torch.cat((baseline_out, row_out, col_out), dim=1)  # (batch, 3, 2000)
        out = self.final_conv(combined)    # (batch, 1, 2000)
        return out
