import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return nn.functional.relu(x + self.res(x))


class GM12878_CVAE_Pro(nn.Module):
    def __init__(self, input_size=64, latent_dim=256, label_dim=3):
        super().__init__()
        self.input_size = input_size

        # 1. Label Embedding (处理特征层)
        self.label_proj = nn.Sequential(
            nn.Linear(label_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU()
        )

        # 2. Encoder: 采用深层残差卷积
        self.enc_init = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)  # 32x32
        self.enc_res = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16x16
            ResidualBlock(128)
        )

        flatten_dim = 128 * (input_size // 4) * (input_size // 4)
        self.fc_mu = nn.Linear(flatten_dim + 128, latent_dim)
        self.fc_logvar = nn.Linear(flatten_dim + 128, latent_dim)

        # 3. Decoder: 多层感知机生成 3D 轨迹
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 128, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Linear(1024, input_size * 3)  # 最终 XYZ 坐标
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        y_feat = self.label_proj(labels)

        # 编码
        h = self.enc_init(x)
        h = self.enc_res(h)
        h = h.view(h.size(0), -1)

        combined = torch.cat([h, y_feat], dim=1)
        mu, logvar = self.fc_mu(combined), self.fc_logvar(combined)

        z = self.reparameterize(mu, logvar)

        # 解码
        pred_coords = self.decoder(torch.cat([z, y_feat], dim=1))
        return pred_coords.view(-1, self.input_size, 3), mu, logvar