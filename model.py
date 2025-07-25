"""
UNet Blocks with Flexible Normalization
---------------------------------------

This module implements:
- double_conv: Two-layer convolution block with normalization.
- downsampling: Downsampling block using strided convolution.
- upsampling: Upsampling block using nearest or transposed convolution.
- RegUNet: A configurable UNet-like architecture for image-to-image tasks.

Adapted from:
https://github.com/anger-man/cascaded-null-space-learning

"""

import torch
import torch.nn as nn

__all__ = ["NORM", "double_conv", "downsampling", "upsampling", "RegUNet"]

def NORM(ch_size: int, normalization: str) -> nn.Module:
    normlayer = nn.ModuleDict({
        'instance': nn.InstanceNorm2d(ch_size),
        'batch': nn.BatchNorm2d(ch_size),
        'none': nn.Identity()
    })
    return normlayer[normalization]


class double_conv(nn.Module):
    def __init__(self, in_f: int, out_f: int, normalization: str = 'batch'):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding='same'),
            NORM(out_f, normalization),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_f, out_f, kernel_size=3, stride=1, padding='same'),
            NORM(out_f, normalization),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class downsampling(nn.Module):
    def __init__(self, in_f: int, out_f: int, normalization: str = 'instance'):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=4, stride=2, padding=1),
            NORM(out_f, normalization),
            nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class upsampling(nn.Module):
    def __init__(self, in_f: int, out_f: int, normalization: str = 'instance', nearest: bool = True):
        super().__init__()

        if nearest:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_f, out_f, kernel_size=4, stride=1, padding='same'),
                NORM(out_f, normalization),
                nn.SiLU()
            )
        else:
            self.up = nn.ConvTranspose2d(in_f, out_f, kernel_size=4, stride=2, padding=1)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding='same'),
            NORM(out_f, normalization),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_f, out_f, kernel_size=3, stride=1, padding='same'),
            NORM(out_f, normalization),
            nn.SiLU(),
            nn.Dropout(p=0.20)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        return self.conv2(x)


class regUnet(nn.Module):
    def __init__(self, n_channels: int, f_size: int, normalization: str = 'batch',
                 out_acti: str = 'tanh', out_channels: int = 16):
        super().__init__()

        # encoder
        self.dc10 = double_conv(n_channels, f_size, normalization)
        self.ds10 = downsampling(f_size, f_size, normalization)
        self.dc11 = double_conv(f_size, 2 * f_size, normalization)
        self.ds11 = downsampling(2 * f_size, 2 * f_size, normalization)
        self.dc12 = double_conv(2 * f_size, 4 * f_size, normalization)
        self.ds12 = downsampling(4 * f_size, 4 * f_size, normalization)
        self.dc13 = double_conv(4 * f_size, 8 * f_size, normalization)
        self.ds13 = downsampling(8 * f_size, 8 * f_size, normalization)
        self.dc14 = double_conv(8 * f_size, 16 * f_size, normalization)

        # decoder
        self.up13 = upsampling(16 * f_size, 8 * f_size, normalization)
        self.up12 = upsampling(8 * f_size, 4 * f_size, normalization)
        self.up11 = upsampling(4 * f_size, 2 * f_size, normalization)
        self.up1_out = upsampling(2 * f_size, f_size, normalization)

        self.out1 = nn.Sequential(
            nn.Conv2d(f_size, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.ModuleDict({
                'relu': nn.ReLU(),
                'tanh': nn.Tanh(),
                'linear': nn.Identity()
            })[out_acti]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        skip10 = self.dc10(x)
        down11 = self.ds10(skip10)
        skip11 = self.dc11(down11)
        down12 = self.ds11(skip11)
        skip12 = self.dc12(down12)
        down13 = self.ds12(skip12)
        skip13 = self.dc13(down13)
        down14 = self.ds13(skip13)
        skip14 = self.dc14(down14)

        upsa13 = self.up13(skip14, skip13)
        upsa12 = self.up12(upsa13, skip12)
        upsa11 = self.up11(upsa12, skip11)
        obranch1 = self.up1_out(upsa11, skip10)

        return self.out1(obranch1)

