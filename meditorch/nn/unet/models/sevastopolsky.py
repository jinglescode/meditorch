import torch
from torch import nn
import torch.nn.functional as F


class DownConv(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.3):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_drop(x)
        x = F.relu(self.conv2(x))
        x = self.conv1_drop(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.3):
        super(UpConv, self).__init__()
        self.up1 = nn.functional.interpolate
        self.downconv = DownConv(in_feat, out_feat, drop_rate)

    def forward(self, x, y):
        x = self.up1(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x


class BottomConv(nn.Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.3):
        super(BottomConv, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_drop(x)
        x = F.relu(self.conv1(x))
        return x



class Sevastopolsky(nn.Module):

    def __init__(self, in_channel, n_classes, drop_rate=0.4):
        super(Sevastopolsky, self).__init__()

        self.conv1 = DownConv(in_channel, 32, drop_rate)
        self.conv2 = DownConv(32, 64, drop_rate)
        self.conv3 = DownConv(64, 64, drop_rate)
        self.conv4 = DownConv(64, 64, drop_rate)

        self.conv5 = BottomConv(64, 64, drop_rate)

        self.conv6 = UpConv(64, 64, drop_rate)
        self.conv7 = UpConv(64, 64, drop_rate)
        self.conv8 = UpConv(64, 32, drop_rate)
        self.conv9 = UpConv(32, 32, drop_rate)

        self.conv10 = nn.Conv2d(32, 1, kernel_size=1, padding=1)

        self.mp = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.mp(x1)

        x2 = self.conv2(x1)
        x2 = self.mp(x2)

        x3 = self.conv3(x2)
        x3 = self.mp(x3)

        x4 = self.conv3(x3)
        x4 = self.mp(x4)

        # Bottom
        x5 = self.conv5(x4)

        # Up-sampling
        x6 = self.conv6(x5, x4) # TODO: RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 1. Got 28 and 14 in dimension 2
        x7 = self.conv7(x6, x3)
        x8 = self.conv8(x7, x2)
        x9 = self.conv9(x8, x1)

        x10 = self.conv10(x9)
        preds = F.sigmoid(x10)

        return preds
