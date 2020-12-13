""" inceptionv3 in pytorch

    Following the implement from https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
[1] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna

    Rethinking the Inception Architecture for Computer Vision
    https://arxiv.org/abs/1512.00567v3
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

# naive inception module
class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        # 1x1 branch
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        # 5x5 branch
        self.branch5x5 = nn.Sequential(
            conv_block(in_channels, 48, kernel_size=1),
            conv_block(48, 64, kernel_size=5, padding=2)
        )

        # double 3x3 branch
        self.branch3x3db = nn.Sequential(
            conv_block(in_channels, 64, kernel_size=1),
            conv_block(64, 96, kernel_size=3, padding=1),
            conv_block(96, 96, kernel_size=3, padding=1),
        )

        # pooling branch
        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, pool_features, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5(x)
        branch3x3db = self.branch3x3db(x)
        branchpool = self.branchpool(x)

        output = torch.cat((branch1x1, branch5x5, branch3x3db, branchpool), dim=1)

        return output

# downsample
# Factorization into smaller convolutions
class InceptionB(nn.Module):
    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        # 3x3 branch
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        # stacked 3x3 branch
        self.branch3x3db = nn.Sequential(
            conv_block(in_channels, 64, kernel_size=1),
            conv_block(64, 96, kernel_size=3, padding=1),
            conv_block(96, 96, kernel_size=3, stride=2),
        )

        # pool branch
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3db = self.branch3x3db(x)
        branchpool = self.branchpool(x)

        output = torch.cat((branch3x3, branch3x3db, branchpool), dim=1)

        return output

# Factorizing Convolutions with Large Filter Size
class InceptionC(nn.Module):
    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

         # 1x1 branch
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7

        # 7x7 branch
        self.branch7x7 = nn.Sequential(
            conv_block(in_channels, c7, kernel_size=1),
            conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        )

        # stacked 7x7 branch
        self.branch7x7db = nn.Sequential(
            conv_block(in_channels, c7, kernel_size=1),
            conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))
        )

        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7(x)
        branch7x7db = self.branch7x7db(x)
        branchpool = self.branchpool(x)

        outputs = [branch1x1, branch7x7, branch7x7db, branchpool]

        return torch.cat(outputs, 1)

# downsample
class InceptionD(nn.Module):
    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        # 3x3 branch
        self.branch3x3 = nn.Sequential(
            conv_block(in_channels, 192, kernel_size=1),
            conv_block(192, 320, kernel_size=3, stride=2)
        )

        # 7x7 branch
        self.branch7x7 = nn.Sequential(
            conv_block(in_channels, 192, kernel_size=1),
            conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            conv_block(192, 192, kernel_size=3, stride=2),
        )

        # pool branch
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch7x7 = self.branch7x7(x)
        branchpool = self.branchpool(x)

        output = torch.cat((branch3x3, branch7x7, branchpool), dim=1)

        return output

# Expand filter bank
class InceptionE(nn.Module):
    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

         # 1x1 branch
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        # 3x3 branch
        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        # stacked 3x3 branch
        self.branch3x3db_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3db_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3db_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3db_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))


        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3db = self.branch3x3db_1(x)
        branch3x3db = self.branch3x3db_2(branch3x3db)
        branch3x3db = [
            self.branch3x3db_3a(branch3x3db),
            self.branch3x3db_3b(branch3x3db)
        ]
        branch3x3db = torch.cat(branch3x3db, 1)

        branchpool = self.branchpool(x)

        outputs = [branch1x1, branch3x3, branch3x3db, branchpool]

        return torch.cat(outputs, 1)

class Inceptionv3(nn.Module):
    def __init__(self, num_classes=100, init_weights=True, aux_logits=False):
        super(Inceptionv3, self).__init__()

        self.aux_logits = aux_logits
        self.prelayers = nn.Sequential(
            # for imagenet 3 x 299 x 299

            # BasicConv2d(3, 32, kernel_size=3, stride=2),
            # BasicConv2d(32, 32, kernel_size=3),
            # BasicConv2d(32, 64, kernel_size=3, padding=1),
            # nn.MaxPool2d(kernel_size=3, stride=2),
            # BasicConv2d(64, 80, kernel_size=1),
            # BasicConv2d(80, 192, kernel_size=3),
            # nn.MaxPool2d(kernel_size=3, stride=2),

            # for cifar100 shape 3 x 32 x 32
            BasicConv2d(3, 32, kernel_size=3, padding=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            BasicConv2d(64, 80, kernel_size=1),
            BasicConv2d(80, 192, kernel_size=3),
        )

        # naive inception block
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        # downsample
        self.Mixed_6a = InceptionB(288)

        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        # downsample
        self.Mixed_7a = InceptionD(768)

        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        if aux_logits:
            self.aux = InceptionAux(768, num_classes)
        else:
            self.aux = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(2048, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.prelayers(x)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)

        if self.aux is not None:
            aux = self.aux(x)

        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048 x 1 x 1
        x = x.view(x.size()[0], -1)
        # N x 2048
        x = self.fc(x)
        # N x num_classes

        if self.aux_logits:
            output = (x, aux)
        output = x

        return output

"""
    Directly copy from
    https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
"""
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d

        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x

