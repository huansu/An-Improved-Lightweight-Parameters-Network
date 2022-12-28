import torch
import torch.nn as nn


# Convolution layers with stride=2
class HeadConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HeadConv, self).__init__()

        self.Conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels),
        )
        self.Conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x1 = self.Conv3x3(x)
        x2 = self.Conv1x1(x)
        x = x1 + x2
        x = self.act(x)

        return x


# Multi-branch layer
class BodyConv(nn.Module):
    def __init__(self, out_channels):
        super(BodyConv, self).__init__()

        self.Conv1x1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.Conv3x3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.Conv5x5 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, groups=out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x1 = self.bn(x)
        x2 = self.Conv1x1(x)
        x3 = self.Conv3x3(x)
        x4 = self.Conv5x5(x)
        x = x1 + x2 + x3 + x4
        x = self.activation(x)

        return x


class Stage_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stage_1, self).__init__()

        # standard convolution with stride = 2
        self.Conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels),
        )
        self.Conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.LeakyReLU()

        # group convolution
        self.NorConv1x1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, groups= out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.NorConv3x3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x1 = self.Conv1x1(x)
        x2 = self.Conv3x3(x)
        x = x1 + x2
        x = self.act(x)

        x1 = self.NorConv1x1(x)
        x2 = self.NorConv3x3(x)
        x = x1 + x2

        return self.act(x)


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, num):
        super(Stage, self).__init__()

        self.HeadConv = HeadConv(in_channels, out_channels)
        self.Circulate = nn.Sequential(
            *[BodyConv(out_channels) for _ in range(num)]
        )

    def forward(self, x):
        x = self.HeadConv(x)
        x = self.Circulate(x)

        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # the input and output channels of different stages
        channels = [3, 64, 128, 256, 512, 1024]

        self.stage_1 = Stage_1(in_channels=channels[0], out_channels=channels[1])
        self.stage_2 = Stage(in_channels=channels[1], out_channels=channels[2], num=2)
        self.stage_3 = Stage(in_channels=channels[2], out_channels=channels[3], num=2)
        self.stage_4 = Stage(in_channels=channels[3], out_channels=channels[4], num=4)
        self.stage_5 = Stage(in_channels=channels[4], out_channels=channels[5], num=2)

    def forward(self, x):
        x1 = self.stage_1(x)

        x2 = self.stage_2(x1)

        x3 = self.stage_3(x2)

        x4 = self.stage_4(x3)

        x5 = self.stage_5(x4)

        return x3, x4, x5
