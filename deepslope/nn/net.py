from torch import nn, concat, abs, Tensor
from torch.nn import functional as F, init

from deepslope.nn.encoder import Encoder
from deepslope.nn.decoder import Decoder


class _Backend(nn.Module):
    def __init__(self, out_features: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(1, out_features, kernel_size=kernel_size,
                              padding='same', padding_mode='replicate')
        self.norm = nn.BatchNorm2d(out_features)
        init.xavier_normal_(self.conv.weight)
        init.uniform_(self.norm.weight)

    def forward(self, x: Tensor):
        return F.leaky_relu(self.norm(self.conv(x)))


class _Frontend(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.conv = nn.Conv2d(in_features, 1, kernel_size=1)
        self.norm = nn.BatchNorm2d(1)
        init.xavier_normal_(self.conv.weight)
        init.uniform_(self.norm.weight)

    def forward(self, x: Tensor):
        return F.gelu(self.norm(self.conv(x)))


class _ResidualBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv_1 = nn.Conv2d(features, features, kernel_size=3,
                                padding='same', padding_mode='zeros', bias=False)
        self.norm_1 = nn.BatchNorm2d(features)
        self.conv_2 = nn.Conv2d(features, features, kernel_size=3,
                                padding='same', padding_mode='zeros', bias=False)
        self.norm_2 = nn.BatchNorm2d(features)

    def forward(self, x: Tensor):
        y = x
        x = F.leaky_relu(self.norm_1(self.conv_1(x)))
        x = F.leaky_relu(self.norm_2(self.conv_2(x)) + y)
        return x


class Net(nn.Module):
    def __init__(self, n: int = 4):
        super().__init__()
        self.backend = _Backend(2 * n, kernel_size=7)
        self.enc1 = Encoder(2 * n, 4 * n, kernel_size=4, stride=4)
        self.enc2 = Encoder(4 * n, 8 * n, kernel_size=2, stride=2)
        self.enc3 = Encoder(8 * n, 16 * n, kernel_size=2, stride=2)
        self.residual_blocks = nn.Sequential(
            _ResidualBlock(n * 16),
            _ResidualBlock(n * 16),
            _ResidualBlock(n * 16),
            _ResidualBlock(n * 16)
        )
        self.dec1 = Decoder(16 * n, 8 * n, kernel_size=2, stride=2, scale=4)
        self.dec2 = Decoder(16 * n, 4 * n, kernel_size=2, stride=2, scale=4)
        self.dec3 = Decoder(8 * n, 2 * n, kernel_size=2, stride=2, scale=8)
        self.rb_2 = nn.Sequential(
            _ResidualBlock(n * 2),
            _ResidualBlock(n * 2),
            _ResidualBlock(n * 2),
            _ResidualBlock(n * 2)
        )
        self.frontend = _Frontend(in_features=4 * n)

    def forward(self, x0: Tensor):
        x0 = self.backend(x0)
        y = self.enc1(x0)
        z = self.enc2(y)
        x = self.enc3(z)
        x = self.residual_blocks(x)
        x = self.dec1(x)
        x = self.dec2(concat((z, x), dim=1))
        x = self.dec3(concat((y, x), dim=1))
        return self.frontend(concat((x0, self.rb_2(x)), dim=1))
