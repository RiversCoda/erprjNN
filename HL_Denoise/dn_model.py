# dn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineTransformer(nn.Module):
    def __init__(self, input_dim=2000, output_dim=2000, nhead=4, num_layers=4, dim_feedforward=512):
        super(BaselineTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=input_dim, nhead=nhead, num_encoder_layers=num_layers, 
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward)
        self.fc_in = nn.Linear(input_dim, input_dim)
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, src, tgt):
        # src and tgt shape: [batch_size, seq_len]
        src = self.fc_in(src).unsqueeze(1)  # [batch_size, 1, seq_len]
        tgt = self.fc_out(tgt).unsqueeze(1)
        output = self.transformer(src, tgt)
        output = output.squeeze(1)
        return output

class RowAttentionTransformer(nn.Module):
    def __init__(self, stft_mode='magnitude', nperseg=64, noverlap=32, nfft=512, nhead=4, num_layers=4, dim_feedforward=512):
        super(RowAttentionTransformer, self).__init__()
        self.stft_mode = stft_mode
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.transformer = nn.Transformer(d_model=128, nhead=nhead, num_encoder_layers=num_layers, 
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward)
        self.fc_out = nn.Linear(128, 2000)

    def forward(self, x):
        # x shape: [batch_size, 2000]
        x = x.numpy()
        stft = torch.stft(torch.tensor(x), n_fft=self.nfft, hop_length=self.nperseg - self.noverlap, 
                          win_length=self.nperseg, return_complex=True)
        # 取0-200Hz部分
        # 假设采样率500Hz，nfft=512，则频率分辨率为500/512 ≈ 0.976 Hz
        # 200Hz对应大约前205个频率
        stft = stft[:, :205, :]
        # 选择行作为token
        if self.stft_mode == 'magnitude':
            spectrogram = torch.abs(stft)
        elif self.stft_mode == 'magnitude_phase':
            spectrogram = torch.cat((torch.abs(stft), torch.angle(stft)), dim=1)
        elif self.stft_mode == 'real_imag':
            spectrogram = torch.cat((stft.real, stft.imag), dim=1)
        else:
            raise ValueError("Unsupported STFT mode.")
        # 行拆分
        tokens = spectrogram.permute(1, 0, 2).reshape(spectrogram.size(1), -1)  # [num_rows, batch_size * features]
        tokens = tokens.unsqueeze(1)  # [num_rows, 1, features]
        output = self.transformer(tokens, tokens)
        output = output.reshape(spectrogram.size(1), -1)
        output = self.fc_out(output)
        # 调整输出尺寸
        output = output[:, :2000]
        return output

class ColumnAttentionTransformer(nn.Module):
    def __init__(self, stft_mode='magnitude', nperseg=64, noverlap=32, nfft=512, nhead=4, num_layers=4, dim_feedforward=512):
        super(ColumnAttentionTransformer, self).__init__()
        self.stft_mode = stft_mode
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.transformer = nn.Transformer(d_model=205, nhead=nhead, num_encoder_layers=num_layers, 
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward)
        self.fc_out = nn.Linear(205, 2000)

    def forward(self, x):
        # 类似RowAttentionTransformer，按列拆分
        x = x.numpy()
        stft = torch.stft(torch.tensor(x), n_fft=self.nfft, hop_length=self.nperseg - self.noverlap, 
                          win_length=self.nperseg, return_complex=True)
        stft = stft[:, :205, :]
        if self.stft_mode == 'magnitude':
            spectrogram = torch.abs(stft)
        elif self.stft_mode == 'magnitude_phase':
            spectrogram = torch.cat((torch.abs(stft), torch.angle(stft)), dim=1)
        elif self.stft_mode == 'real_imag':
            spectrogram = torch.cat((stft.real, stft.imag), dim=1)
        else:
            raise ValueError("Unsupported STFT mode.")
        # 列拆分
        tokens = spectrogram.permute(2, 0, 1).reshape(spectrogram.size(2), -1)  # [num_cols, batch_size * features]
        tokens = tokens.unsqueeze(1)  # [num_cols, 1, features]
        output = self.transformer(tokens, tokens)
        output = output.reshape(spectrogram.size(2), -1)
        output = self.fc_out(output)
        output = output[:, :2000]
        return output

class CombinationAttentionTransformer(nn.Module):
    def __init__(self, stft_mode='magnitude', nperseg=64, noverlap=32, nfft=512, nhead=4, num_layers=4, dim_feedforward=512):
        super(CombinationAttentionTransformer, self).__init__()
        self.row_transformer = RowAttentionTransformer(stft_mode, nperseg, noverlap, nfft, nhead, num_layers, dim_feedforward)
        self.column_transformer = ColumnAttentionTransformer(stft_mode, nperseg, noverlap, nfft, nhead, num_layers, dim_feedforward)
        self.fc = nn.Linear(4000, 2000)

    def forward(self, x):
        row_out = self.row_transformer(x)
        column_out = self.column_transformer(x)
        combined = torch.cat((row_out, column_out), dim=1)
        output = self.fc(combined)
        return output

class MultiCombinationTransformer(nn.Module):
    def __init__(self, stft_mode='magnitude', nperseg=64, noverlap=32, nfft=512, nhead=4, num_layers=4, dim_feedforward=512):
        super(MultiCombinationTransformer, self).__init__()
        self.baseline = BaselineTransformer(nhead=nhead, num_layers=num_layers, dim_feedforward=dim_feedforward)
        self.row_transformer = RowAttentionTransformer(stft_mode, nperseg, noverlap, nfft, nhead, num_layers, dim_feedforward)
        self.column_transformer = ColumnAttentionTransformer(stft_mode, nperseg, noverlap, nfft, nhead, num_layers, dim_feedforward)
        self.fc = nn.Linear(6000, 2000)

    def forward(self, x):
        baseline_out = self.baseline(x, x)
        row_out = self.row_transformer(x)
        column_out = self.column_transformer(x)
        combined = torch.cat((baseline_out, row_out, column_out), dim=1)
        output = self.fc(combined)
        return output

# dn_model.py

class SimpleDenoisingModel(nn.Module):
    def __init__(self, input_dim=2000, output_dim=2000):
        super(SimpleDenoisingModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
