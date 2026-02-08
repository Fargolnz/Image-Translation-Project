import torch
import torch.nn as nn

class DiscBlock(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, stride, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block1 = DiscBlock(64, 128, stride=2)
        self.block2 = DiscBlock(128, 256, stride=2)
        self.block3 = DiscBlock(256, 512, stride=1)

        self.final = nn.Conv2d(512, 1, 4, 1, 1)


    def forward(self, x, y):
        # x: corrupted image
        # y: real or generated image
        input = torch.cat([x, y], dim=1)
        out = self.initial(input)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        return self.final(out)
