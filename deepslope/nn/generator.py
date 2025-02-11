from torch import nn, Tensor, concat
from torch.nn import init, functional as F

from deepslope.nn.decoder import Decoder
from deepslope.nn.residual import ResidualBlock


class _Encoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features,
                              kernel_size=kernel_size, stride=stride, bias=False)
        self.norm = nn.BatchNorm2d(out_features)
        init.xavier_normal_(self.conv.weight)
        init.uniform_(self.norm.weight)

    def forward(self, x: Tensor):
        x = self.conv(x)
        x = self.norm(x)
        return F.gelu(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            _Encoder(1, 16, kernel_size=5),
            _Encoder(16, 16, kernel_size=5),
            _Encoder(16, 16, kernel_size=5),
            _Encoder(16, 16, kernel_size=5),
            _Encoder(16, 16, kernel_size=5),
            _Encoder(16, 16, kernel_size=5),
            _Encoder(16, 32, kernel_size=5),
            _Encoder(32, 1, kernel_size=5)
        )

    def forward(self, x: Tensor, noise: Tensor):
        x = self.layers(noise)
        return x
