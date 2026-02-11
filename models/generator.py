import torch
import torch.nn as nn

# ---------- Down Block ----------
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)
        ]

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ---------- Up Block ----------
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            layers.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ---------- Generator (U-Net 8 layers) ----------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Downsampling
        self.down1 = DownBlock(3, 64, normalize=False)    # 256 → 128
        self.down2 = DownBlock(64, 128)                   # 128 → 64
        self.down3 = DownBlock(128, 256)                  # 64 → 32
        self.down4 = DownBlock(256, 512)                  # 32 → 16
        self.down5 = DownBlock(512, 512)                  # 16 → 8
        self.down6 = DownBlock(512, 512)                  # 8 → 4
        self.down7 = DownBlock(512, 512)                  # 4 → 2
        self.down8 = DownBlock(512, 512, normalize=False) # 2 → 1

        # Upsampling
        self.up1 = UpBlock(512, 512, dropout=True)
        self.up2 = UpBlock(1024, 512, dropout=True)
        self.up3 = UpBlock(1024, 512, dropout=True)
        self.up4 = UpBlock(1024, 512)
        self.up5 = UpBlock(1024, 256)
        self.up6 = UpBlock(512, 128)
        self.up7 = UpBlock(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):

        # Down path
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Up path with skip connections
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))

        out = self.final(torch.cat([u7, d1], 1))

        return out