import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True, use_act=True):
        super().__init__()

        if down:
            self.conv = nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False)

        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.d1 = DownBlock(3, 64, normalize=False)
        self.d2 = DownBlock(64, 128)
        self.d3 = DownBlock(128, 256)
        self.d4 = DownBlock(256, 512)
        self.d5 = DownBlock(512, 512)
        self.d6 = DownBlock(512, 512)
        self.d7 = DownBlock(512, 512)
        self.d8 = DownBlock(512, 512, normalize=False)

        # Decoder (🔥 دقت کن به in_channels)
        self.u1 = UpBlock(512, 512, dropout=True)
        self.u2 = UpBlock(1024, 512, dropout=True)
        self.u3 = UpBlock(1024, 512, dropout=True)
        self.u4 = UpBlock(1024, 512)
        self.u5 = UpBlock(1024, 256)
        self.u6 = UpBlock(512, 128)
        self.u7 = UpBlock(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8)
        u2 = self.u2(torch.cat([u1, d7], dim=1))
        u3 = self.u3(torch.cat([u2, d6], dim=1))
        u4 = self.u4(torch.cat([u3, d5], dim=1))
        u5 = self.u5(torch.cat([u4, d4], dim=1))
        u6 = self.u6(torch.cat([u5, d3], dim=1))
        u7 = self.u7(torch.cat([u6, d2], dim=1))

        return self.final(torch.cat([u7, d1], dim=1))

"""model = Generator()
x = torch.randn(1, 3, 256, 256)
y = model(x)
print(y.shape)"""
