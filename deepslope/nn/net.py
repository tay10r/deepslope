from torch import nn, concat, Tensor
from torch.nn import functional as F

from deepslope.nn.encoder import Encoder
from deepslope.nn.decoder import Decoder


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = Encoder(1, 64, kernel_size=7)
        self.enc2 = Encoder(64, 128, kernel_size=5)
        self.dec1 = Decoder(128, 64, kernel_size=5)
        self.dec2 = Decoder(128, 16, kernel_size=5)
        self.__frontend = nn.Sequential(
            nn.Conv2d(16, 1, 5, padding='same'),
            nn.BatchNorm2d(1)
        )

    def forward(self, x: Tensor):
        y = self.enc1(x)
        x = self.enc2(y)
        x = self.dec1(x)
        x = self.dec2(concat((y, x), dim=1))
        return F.sigmoid(self.__frontend(x))
