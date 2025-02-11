from torch import nn, Tensor, concat
from torch.nn import functional as F, init

from deepslope.nn.residual import ResidualBlock


class _Encoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size_1: int = 3, kernel_size_2: int = 3):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_features,
                                out_features,
                                kernel_size=kernel_size_1,
                                stride=2)
        self.conv_2 = nn.Conv2d(out_features,
                                out_features,
                                kernel_size=kernel_size_2)
        self.norm = nn.BatchNorm2d(out_features)
        init.xavier_normal_(self.conv_1.weight)
        init.xavier_normal_(self.conv_2.weight)
        init.uniform_(self.norm.weight)

    def forward(self, x: Tensor):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.norm(x)
        return F.leaky_relu(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # initial input size is 256x256
        self.conv_layers = nn.Sequential(
            # 256 -> (256 - 4) / 2 = 126 -> 122
            _Encoder(1, 8, kernel_size_1=5, kernel_size_2=5),
            # 122 -> (122 - 2) / 2 = 60 -> 58
            _Encoder(8, 16),
            # 58 -> (58 - 2) / 2 = 28 -> 26
            _Encoder(16, 24),
            # 26 -> (26 - 2) / 2 = 12 -> 10
            _Encoder(24, 1),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(100, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x: Tensor):
        x = self.conv_layers(x)
        x = self.linear_layers(x.view(x.size(0), -1))
        return x
