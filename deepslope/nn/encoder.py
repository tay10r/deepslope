from torch import nn, Tensor
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features,
                              kernel_size, padding='same')
        self.norm = nn.BatchNorm2d(out_features)
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.uniform_(self.norm.weight)

    def forward(self, x: Tensor):
        return F.relu(F.max_pool2d(self.norm(self.conv(x)), kernel_size=2))
