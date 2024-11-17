import torch
import torch.nn as nn

# 定义ResNet块
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad1d(1),
            nn.Conv1d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm1d(dim),
            nn.ReLU(True),

            nn.ReflectionPad1d(1),
            nn.Conv1d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm1d(dim)
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

# 定义ResNet生成器
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_blocks=6):
        super(ResnetGenerator, self).__init__()
        model = [
            nn.ReflectionPad1d(3),
            nn.Conv1d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm1d(ngf),
            nn.ReLU(True)
        ]

        # 下采样
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv1d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm1d(ngf * mult * 2),
                nn.ReLU(True)
            ]

        # 残差块
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]

        # 上采样
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose1d(ngf * mult, int(ngf * mult / 2),
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm1d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        model += [
            nn.ReflectionPad1d(3),
            nn.Conv1d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# 定义PatchGAN判别器
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        kw = 4
        padw = 1

        sequence = [
            nn.Conv1d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.InstanceNorm1d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm1d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv1d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

# 定义位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数项
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数项
        pe = pe.unsqueeze(1)  # 形状 (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x形状: (seq_len, batch_size, d_model)
        seq_len = x.size(0)
        pe = self.pe[:seq_len]  # 形状 (seq_len, 1, d_model)
        return x + pe  # 广播到 batch_size

# 定义Transformer生成器
class TransformerGenerator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, d_model=128, nhead=8, num_layers=8):
        super(TransformerGenerator, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # 线性层将输入映射到d_model维度
        self.input_proj = nn.Linear(self.input_channels, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 线性层将d_model映射回输出通道数
        self.output_proj = nn.Linear(d_model, self.output_channels)

        # Tanh激活函数
        self.tanh = nn.Tanh()
    def forward(self, x):
        # x形状: (batch_size, input_channels, signal_length)
        batch_size, input_channels, signal_length = x.size()

        # 转换x为 (batch_size, signal_length, input_channels)
        x = x.permute(0, 2, 1)

        # 映射输入通道到d_model
        x = self.input_proj(x)
        # x形状: (batch_size, signal_length, d_model)

        # 转置为 Transformer 需要的形状: (signal_length, batch_size, d_model)
        x = x.permute(1, 0, 2)

        # 添加位置编码
        x = self.pos_encoder(x)

        # 通过Transformer编码器
        x = self.transformer_encoder(x)

        # 转换回 (batch_size, signal_length, d_model)
        x = x.permute(1, 0, 2)

        # 映射回输出通道数
        x = self.output_proj(x)
        # x形状: (batch_size, signal_length, output_channels)

        # 转置回 (batch_size, output_channels, signal_length)
        x = x.permute(0, 2, 1)

        # 应用 Tanh 激活函数
        x = self.tanh(x)

        return x
