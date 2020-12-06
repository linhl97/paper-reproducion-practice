"""vgg in pytorch

[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""

import torch
import torch.nn as nn

cfg = {
    'A' :[64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],

}
class VGG(nn.Module):
    def __init__(self, features, num_class=100, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        # For cifar100, the input image's size is 32, after five maxpooling
        # spatial size of feature size become 1x1. So it doesn't need add average pooling.
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        output = self.features(x)
        output = output.view(x.size()[0], -1)
        output = self.classifier(output)

        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m , nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        else:
            layers += [nn.Conv2d(in_channels, l, kernel_size=3, padding=1)]
            if batch_norm:
                layers += [nn.BatchNorm2d(l), nn.ReLU(inplace=True)]
            else:
                layers += [nn.ReLU(inplace=True)]
            in_channels = l

    return nn.Sequential(*layers)

def vgg11_bn(num_class=100, init_weights=True):
    return VGG(make_layers(cfg['A'], batch_norm=True), num_class, init_weights)

def vgg13_bn(num_class=100, init_weights=True):
    return VGG(make_layers(cfg['B'], batch_norm=True), num_class, init_weights)

def vgg16_bn(num_class=100, init_weights=True):
    return VGG(make_layers(cfg['D'], batch_norm=True), num_class, init_weights)

def vgg19_bn(num_class=100, init_weights=True):
    return VGG(make_layers(cfg['E'], batch_norm=True), num_class, init_weights)