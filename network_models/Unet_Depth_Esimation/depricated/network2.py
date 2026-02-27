import numpy as np

import torch, torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# U-Net (compact)
# ----------------------------

class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
            nn.Conv2d(o,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
        )
    def forward(self,x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, ch=[64,128,256,512,1024]):
        super().__init__()
        self.d1 = DoubleConv(3, ch[0]);  self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(ch[0], ch[1]);  self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(ch[1], ch[2]);  self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(ch[2], ch[3]);  self.p4 = nn.MaxPool2d(2)
        self.b  = DoubleConv(ch[3], ch[4])

        self.u4  = nn.ConvTranspose2d(ch[4], ch[3], 2, 2)
        self.u4d = DoubleConv(ch[4], ch[3])

        self.u3  = nn.ConvTranspose2d(ch[3], ch[2], 2, 2)
        self.u3d = DoubleConv(ch[3], ch[2])

        self.u2  = nn.ConvTranspose2d(ch[2], ch[1], 2, 2)
        self.u2d = DoubleConv(ch[2], ch[1])

        self.u1  = nn.ConvTranspose2d(ch[1], ch[0], 2, 2)
        self.u1d = DoubleConv(ch[0] + ch[0], ch[0])

        self.out = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        d3 = self.d3(self.p2(d2))
        d4 = self.d4(self.p3(d3))
        b  = self.b(self.p4(d4))

        x = self.u4(b); x = self.u4d(torch.cat([x, d4], 1))
        x = self.u3(x); x = self.u3d(torch.cat([x, d3], 1))
        x = self.u2(x); x = self.u2d(torch.cat([x, d2], 1))
        x = self.u1(x); x = self.u1d(torch.cat([x, d1], 1))
        return self.out(x)  # log-depth

