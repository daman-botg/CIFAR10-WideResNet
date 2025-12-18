import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def conv2d(ni, nf, stride=1, ks=3):
    return nn.Conv2d(ni, nf, ks, stride, ks//2, bias=False)

def batchnrelu(ni, nf):
    return nn.Sequential(
        nn.BatchNorm2d(ni),
        nn.ReLU(inplace=True),
        conv2d(ni, nf)
    )

class ResidualBlock(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(ni)
        self.conv1 = conv2d(ni, nf, stride)
        self.conv2 = batchnrelu(nf, nf)

        self.shortcut = nn.Identity()
        if ni != nf or stride != 1:
            self.shortcut = conv2d(ni, nf, stride=stride, ks=1)

    def forward(self, x):
        x_in = F.relu(self.bn(x), inplace=True)
        r = self.shortcut(x_in)
        x_out = self.conv1(x_in)
        x_out = self.conv2(x_out) * 0.2
        return x_out + r

def makegroup(N, ni, nf, stride):
    layers = [ResidualBlock(ni, nf, stride)]
    for _ in range(1, N):
        layers.append(ResidualBlock(nf, nf))
    return layers

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class WideResNet(nn.Module):
    def __init__(self, ngroups, N, nclasses, k=6, nstart=16):
        super().__init__()
        layers = [conv2d(3, nstart)]
        nchannels = [nstart]

        for i in range(ngroups):
            nchannels.append(nstart * (2**i) * k)
            stride = 2 if i > 0 else 1
            layers += makegroup(N, nchannels[i], nchannels[i+1], stride)

        layers += [
            nn.BatchNorm2d(nchannels[ngroups]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(nchannels[ngroups], nclasses)
        ]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)

def wrn22():
    return WideResNet(ngroups=3, N=3, nclasses=10)