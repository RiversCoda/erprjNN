import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):  # max_len can be adjusted
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(1)  # shape (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        seq_len = x.size(0)
        pe = self.pe[:seq_len]  # shape (seq_len, 1, d_model)
        return x + pe  # broadcasting over batch_size

class MultiLayerMultiHeadTransformer(nn.Module):
    def __init__(self, n_layers=8, n_heads=8, d_model=128):
        super(MultiLayerMultiHeadTransformer, self).__init__()
        
        self.input_channels = 1
        self.output_channels = 1
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        
        # Linear layer to map from input_channels to d_model
        self.input_proj = nn.Linear(self.input_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Linear layer to map from d_model to output_channels
        self.output_proj = nn.Linear(d_model, self.output_channels)
        
        # Tanh activation function
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        # x shape: (batch_size, input_channels, signal_length)
        batch_size, input_channels, signal_length = x.size()
        
        # Transpose x to (batch_size, signal_length, input_channels)
        x = x.permute(0, 2, 1)
        
        # Map input_channels to d_model
        x = self.input_proj(x)
        # x shape: (batch_size, signal_length, d_model)
        
        # Transpose for transformer: (signal_length, batch_size, d_model)
        x = x.permute(1, 0, 2)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        
        # Transpose back to (batch_size, signal_length, d_model)
        x = x.permute(1, 0, 2)
        
        # Map from d_model to output_channels
        x = self.output_proj(x)
        # x shape: (batch_size, signal_length, output_channels)
        
        # Transpose back to (batch_size, output_channels, signal_length)
        x = x.permute(0, 2, 1)
        
        # Apply Tanh activation
        x = self.tanh(x)
        
        return x

class UNetGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UNetGenerator, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2)
        )
        self.enc5 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2)
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(512, 512, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(1024, 256, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(512, 128, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(256, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose1d(128, output_channels, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )
        

    def forward(self, x):
        # Encoding path
        e1 = self.enc1(x)
        print(f"e1 shape: {e1.shape}")
        e2 = self.enc2(e1)
        print(f"e2 shape: {e2.shape}")
        e3 = self.enc3(e2)
        print(f"e3 shape: {e3.shape}")
        e4 = self.enc4(e3)
        print(f"e4 shape: {e4.shape}")
        e5 = self.enc5(e4)
        print(f"e5 shape: {e5.shape}")

        # Decoding path with skip connections
        d1 = self.dec1(e5)
        print(f"d1 shape before concat: {d1.shape}")
        print(f"e4 shape: {e4.shape}")
        d1 = torch.cat([d1, e4], dim=1)
        print(f"d1 shape after concat: {d1.shape}")

        d2 = self.dec2(d1)
        print(f"d2 shape before concat: {d2.shape}")
        print(f"e3 shape: {e3.shape}")
        d2 = torch.cat([d2, e3], dim=1)
        print(f"d2 shape after concat: {d2.shape}")

        d3 = self.dec3(d2)
        print(f"d3 shape before concat: {d3.shape}")
        print(f"e2 shape: {e2.shape}")
        d3 = torch.cat([d3, e2], dim=1)
        print(f"d3 shape after concat: {d3.shape}")

        d4 = self.dec4(d3)
        print(f"d4 shape before concat: {d4.shape}")
        print(f"e1 shape: {e1.shape}")
        d4 = torch.cat([d4, e1], dim=1)
        print(f"d4 shape after concat: {d4.shape}")

        out = self.dec5(d4)
        print(f"out shape: {out.shape}")
        return out

class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # 通道数减半
            nn.LeakyReLU(0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),  # 通道数减半
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),  # 通道数减半
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 1, kernel_size=4, stride=1, padding=1),  # 减少一层卷积
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# class Discriminator(nn.Module):
#     def __init__(self, input_channels=1):
#         super(Discriminator, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv1d(input_channels, 64, kernel_size=4, stride=2, padding=1),
#             nn.LeakyReLU(0.2)
#         )
#         self.layer2 = nn.Sequential(
#             nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(0.2)
#         )
#         self.layer3 = nn.Sequential(
#             nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(0.2)
#         )
#         self.layer4 = nn.Sequential(
#             nn.Conv1d(256, 512, kernel_size=4, stride=1, padding=1),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(0.2)
#         )
#         self.layer5 = nn.Sequential(
#             nn.Conv1d(512, 1, kernel_size=4, stride=1, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         return x
 