"""google net in pytorch
[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
    Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x

class Inception(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj, conv_block=None):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        # 1x1 branch
        self.branch1 = conv_block(in_channels, n1x1, kernel_size=1)

        # 3x3 branch
        self.branch2 = nn.Sequential(
            conv_block(in_channels, n3x3_reduce, kernel_size=1),
            conv_block(n3x3_reduce, n3x3, kernel_size=3, padding=1)
        )

        # 5x5 branch
        self.branch3 = nn.Sequential(
            conv_block(in_channels, n5x5_reduce, kernel_size=1),
            conv_block(n5x5_reduce, n5x5, kernel_size=5, padding=2)
        )

        # pooling brance
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        output = torch.cat((branch1, branch2, branch3, branch4), dim=1)

        return output

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_class, conv_block):
        if conv_block is None:
            conv_block = BasicConv2d

        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_class)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = x.view(x.size()[0], -1)
        # N x 2048
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        # N x 1024
        x = F.dropout(x, p=0.7)
        x = self.fc2(x)
        # N x num_classes

        return x

class GoogLeNet(nn.Module):
    def __init__(self, num_class=100, init_weights=True, aux_logits=False):
        super(GoogLeNet, self).__init__()

        self.aux_logits = aux_logits
        self.prelayers = nn.Sequential(
            # for imagenet 3 x 224 x 224

            # BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3),
            # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            # BasicConv2d(64, 64, kernel_size=1),
            # BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1),
            # nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

            # for cifar100 shape 3 x 32 x 32
            BasicConv2d(3, 64, kernel_size=3, stride=1, padding=1),
            BasicConv2d(64, 192, kernel_size=3, stride=1, padding=1)

        )

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(512, 1000)
            self.aux2 = InceptionAux(528, 1000)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, 100)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.prelayers(x)
        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        if self.aux1 is not None:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if self.aux2 is not None:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)

        # N x 1024 x 7 x 7
        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        # N x 1024
        x = self.fc(x)
        # N x num_classes

        if self.aux_logits:
            output = (x, aux1, aux2)
        output = x

        return output

