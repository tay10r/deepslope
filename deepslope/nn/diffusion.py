from torch import nn, Tensor, concat
from torch.nn import functional as F


class _ResBlock(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            features, features, kernel_size=3, padding=1, bias=False)
        self.norm_1 = nn.BatchNorm2d(features)
        self.conv_2 = nn.Conv2d(
            features, features, kernel_size=3, padding=1, bias=False)
        self.norm_2 = nn.BatchNorm2d(features)
        nn.init.xavier_normal_(self.conv_1.weight)
        nn.init.xavier_normal_(self.conv_2.weight)
        nn.init.uniform_(self.norm_1.weight)
        nn.init.uniform_(self.norm_2.weight)

    def forward(self, x: Tensor):
        y = x
        x = F.silu(self.norm_1(self.conv_1(x)))
        x = F.silu(self.norm_2(self.conv_2(x)) + y)
        return x


class _Encoder(nn.Module):
    def __init__(self, in_features: int, out_features: int, res_blocks: int):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features,
                              kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_features)
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.uniform_(self.norm.weight)
        r: list[nn.Module] = []
        for i in range(res_blocks):
            r.append(_ResBlock(out_features))
        self.res_blocks = nn.Sequential(*r)

    def forward(self, x: Tensor):
        x = self.conv(x)
        x = self.norm(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.silu(x)
        x = self.res_blocks(x)
        return x


class _Decoder(nn.Module):
    def __init__(self, in_features, out_features: int, res_blocks: int):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features,
                              kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_features)
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.uniform_(self.norm.weight)
        r: list[nn.Module] = []
        for i in range(res_blocks):
            r.append(_ResBlock(out_features))
        self.res_blocks = nn.Sequential(*r)

    def forward(self, x: Tensor):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv(x)
        x = self.norm(x)
        x = F.silu(x)
        x = self.res_blocks(x)
        return x


class DiffusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 256 -> 128
        self.enc1 = _Encoder(1, 64, res_blocks=2)
        # 128 -> 64
        self.enc2 = _Encoder(64, 128, res_blocks=2)
        # 64 -> 32
        self.enc3 = _Encoder(128, 256, res_blocks=2)
        # 32 -> 16
        self.enc4 = _Encoder(256, 512, res_blocks=4)
        # 16 -> 32
        self.dec4 = _Decoder(512, 256, res_blocks=4)
        # 32 -> 64
        self.dec3 = _Decoder(512, 128, res_blocks=2)
        # 64 -> 128
        self.dec2 = _Decoder(256, 64, res_blocks=2)
        # 128 -> 256
        self.dec1 = _Decoder(128, 32, res_blocks=8)
        self.last_conv = nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x: Tensor):
        x = self.enc1(x)
        y = self.enc2(x)
        z = self.enc3(y)
        w = self.enc4(z)
        w = self.dec4(w)
        z = self.dec3(concat((w, z), dim=1))
        y = self.dec2(concat((z, y), dim=1))
        x = self.dec1(concat((y, x), dim=1))
        x = self.last_conv(x)
        return x
