"""densenet in pytorch

[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.
    Densely Connected Convolutional Networks
    https://arxiv.org/abs/1608.06993v5

"""

import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    """Bottleneck DenseLayer
    """
    def __init__(self, in_channels, growth_rate, bn_size, drop_rate):
        super(Bottleneck, self).__init__()

        inner_channel = bn_size * growth_rate
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.drop_rate = float(drop_rate)

    def forward(self, x):
        new_features = self.bottleneck(x)
        if self.drop_rate > 0:
            new_features = nn.Dropout(new_features, p=self.drop_rate)

        return torch.cat([x, new_features], 1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.downsample(x)

class DenseBlock(nn.Module):

    def __init__(self, in_channels, num_layers, growth_rate, bn_size, drop_rate):
        super(DenseBlock, self).__init__()

        self.denseblock = nn.Sequential()
        for i in range(num_layers):
            layer = Bottleneck(in_channels + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.denseblock.add_module('denselayer%d' % (i+1), layer)

    def forward(self, x):
        out = self.denseblock(x)

        return out

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    """Densenet-BC model class

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        reduction (float) - compression factor in transition layer
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, block_config, bn_size=4, growth_rate=12, reduction=0.5, drop_rate=0, num_class=100, init_weights=True):
        super(DenseNet, self).__init__()

        inner_channels = 2 * growth_rate

        self.features = nn.Sequential(
            ## for Imagenet 224x224
            # nn.Conv2d(3, inner_channels, kernel_size=7, padding=3, stride=2, bias=False),
            # nn.BatchNorm2d(inner_channels),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
            ## for Cifar 32x32
            nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, stride=1, bias=False)
        )

        num_features = inner_channels
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_features, num_layers, growth_rate, bn_size, drop_rate)
            self.features.add_module('denseblock%d'%(i+1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                out_features = int(num_features * reduction)
                trans = Transition(num_features, out_features)
                self.features.add_module('transition%d'%(i+1), trans)
                num_features = out_features

        self.features.add_module('bn', nn.BatchNorm2d(num_features))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(num_features, num_class)


        if init_weights:
            # Official init from torch repo.
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size()[0], -1)
        out = self.linear(out)

        return out

def densenet121():
    return DenseNet([6,12,24,16], growth_rate=32)

def densenet161():
    return DenseNet([6,12,36,24], growth_rate=48)

def densenet169():
    return DenseNet([6,12,32,32], growth_rate=32)

def densenet201():
    return DenseNet([6,12,48,32], growth_rate=32)

