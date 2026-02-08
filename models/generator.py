import torch
import torch.nn as nn

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
    def __init__(self, in_channels=3):
        super().__init__()

        # Encoder
        self.down1 = ConvBlock(in_channels, 64, down=True, use_act=False)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.down4 = ConvBlock(256, 512)
        self.down5 = ConvBlock(512, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.ReLU()
        )

        # Decoder
        self.up1 = ConvBlock(512, 512, down=False)
        self.up2 = ConvBlock(1024, 256, down=False)
        self.up3 = ConvBlock(512, 128, down=False)
        self.up4 = ConvBlock(256, 64, down=False)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        bottleneck = self.bottleneck(d5)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d5], dim=1))
        up3 = self.up3(torch.cat([up2, d4], dim=1))
        up4 = self.up4(torch.cat([up3, d3], dim=1))

        return self.final(torch.cat([up4, d2], dim=1))
