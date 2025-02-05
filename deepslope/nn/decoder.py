from torch import nn, Tensor
from torch.nn import functional as F


class Decoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int, stride: int, scale: int):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride)
        self.norm = nn.BatchNorm2d(out_features)
        self.scale = scale
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.uniform_(self.norm.weight)

    def forward(self, x: Tensor) -> Tensor:
        return F.leaky_relu(self.norm(self.conv(F.interpolate(x, scale_factor=self.scale, mode='bilinear'))))
