"""wideresnet in pytorch

[1] Sergey Zagoruyko, Nikos Komodakis

    Wide Residual Networks
    https://arxiv.org/abs/1605.07146
"""

import torch
import torch.nn as nn

class WideBasic(nn.Module):


    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=None):
        super(WideBasic, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # pre activation and dropout
        self.bn1 = norm_layer(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn2 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.dropout = nn.Dropout()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.dropout(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity

        return out

class WideResNet(nn.Module):

    def __init__(self, block, num_classes=100, norm_layer=None, init_weights=True, depth=40, widen_factor=1):
        super(WideResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = 16
        assert ((depth-4) % 6 ==0), 'Wide-resnet depth should be 6n+4'
        n = int((depth - 4) / 6)
        k = widen_factor

        self.conv1 = nn.Sequential(
            # for 3 x 224 x 224
            # nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            # norm_layer(self.in_channels),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            # for cifar100 3 x 32 x 32
            nn.Conv2d(3, self.in_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(self.in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = self._make_layer(block, 16 * k, n, 1)
        self.conv3 = self._make_layer(block, 32 * k, n, 2)
        self.conv4 = self._make_layer(block, 64 * k, n, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * k, num_classes)

        if init_weights:
            self._initialize_weights()

    def _make_layer(self, block, out_channels, num_block, stride):
        norm_layer = self._norm_layer
        downsample = None

        if stride !=1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride),
                norm_layer(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample, norm_layer))
        self.in_channels = out_channels

        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m , nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.avg_pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


def wideresnet(depth=40, widen_factor=10):
    """ return a WiderResnet-40-10 object
    """
    return WideResNet(WideBasic, depth=depth, widen_factor=widen_factor)

