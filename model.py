import torch
import torch.nn as nn

def NORM(ch_size, normalization):
    normlayer = nn.ModuleDict([
        ['instance', nn.InstanceNorm2d(ch_size)],
        ['batch', nn.BatchNorm2d(ch_size)],
        ['none', nn.Identity()]
    ])
    return normlayer[normalization]

class DoubleConv(nn.Module):
    def __init__(self, in_f, out_f, normalization='batch'):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_f, out_f, kernel_size=3, stride=1, padding="same"),
            NORM(out_f, normalization),
            nn.SiLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_f, out_f, kernel_size=3, stride=1, padding="same"),
            NORM(out_f, normalization),
            nn.SiLU()
        )
        
    def forward(self, x):
        return self.conv2(self.conv1(x))

class regUnet(nn.Module):
    def __init__(self, n_channels=1, f_size=32, normalization='batch', out_channels=16):
        super(regUnet, self).__init__()

        self.dc10 = DoubleConv(n_channels, f_size, normalization)
        self.ds10 = nn.Conv2d(f_size, f_size, kernel_size=4, stride=2, padding=1)
        self.dc11 = DoubleConv(f_size, 2 * f_size, normalization)
        self.ds11 = nn.Conv2d(2 * f_size, 2 * f_size, kernel_size=4, stride=2, padding=1)
        self.dc12 = DoubleConv(2 * f_size, 4 * f_size, normalization)
        self.ds12 = nn.Conv2d(4 * f_size, 4 * f_size, kernel_size=4, stride=2, padding=1)
        
        self.out = nn.Sequential(
            nn.Conv2d(f_size, out_channels, 3, stride=1, padding="same"),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.float()
        skip1 = self.dc10(x)
        x = self.ds10(skip1)
        skip2 = self.dc11(x)
        x = self.ds11(skip2)
        x = self.dc12(x)
        return self.out(x)

