from torch import nn, Tensor
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, features: int, same_padding: bool = True):
        super().__init__()
        self.same_padding = same_padding
        padding = 'same' if same_padding else 0
        self.conv_1 = nn.Conv2d(
            features, features, kernel_size=3, padding=padding, bias=False)
        self.norm_1 = nn.BatchNorm2d(features)
        self.conv_2 = nn.Conv2d(
            features, features, kernel_size=3, padding=padding, bias=False)
        self.norm_2 = nn.BatchNorm2d(features)

    def forward(self, x: Tensor):
        y = x
        if not self.same_padding:
            y = y[:, :, 2:-2, 2:-2]
        x = F.leaky_relu(self.norm_1(self.conv_1(x)))
        x = F.leaky_relu(self.norm_2(self.conv_2(x)) + y)
        return x
