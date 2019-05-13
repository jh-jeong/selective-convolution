from collections import OrderedDict
import math

import torch
import torch.nn as nn

from selective_convolution import SelectiveConv2d
from models import BRC


class PreBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, args):
        super(PreBottleneck, self).__init__()

        self.in_planes = in_planes
        self.planes = planes
        self.add_planes = planes * self.expansion - in_planes

        if args.get('use_sconv', False):
            gamma = args['gamma']
            K = args.get('K', 3)
            N_max = args.get('N_max', None)

            self.brc_1 = SelectiveConv2d(in_planes, planes, kernel_size=1,
                                         gamma=gamma, K=K, N_max=N_max)
            self.brc_2 = SelectiveConv2d(planes, planes, kernel_size=3, padding=1,
                                         gamma=gamma, K=0, N_max=N_max)
            self.brc_3 = SelectiveConv2d(planes, self.expansion * planes, kernel_size=1,
                                         gamma=gamma, K=K, N_max=N_max)
        else:
            self.brc_1 = BRC(in_planes, planes, kernel_size=1)
            self.brc_2 = BRC(planes, planes, kernel_size=3, padding=1)
            self.brc_3 = BRC(planes, self.expansion * planes, kernel_size=1)

    def forward(self, x):
        x_ = self.brc_1(x)
        x_ = self.brc_2(x_)
        x_ = self.brc_3(x_)

        if self.add_planes > 0:
            N, _, H, W = x_.size()
            padding = x_.new_zeros(N, self.add_planes, H, W)
            x = torch.cat((x, padding), 1)

        out = x + x_
        return out


class _ResidualBlock(nn.Sequential):
    def __init__(self, args, n_layers, in_channels, planes):
        super(_ResidualBlock, self).__init__()

        layer = PreBottleneck(in_channels, planes, args)
        self.add_module('layer1', layer)

        expansion = PreBottleneck.expansion

        for i in range(n_layers-1):
            layer = PreBottleneck(expansion*planes, planes, args)
            self.add_module('layer%d' % (i + 2), layer)


class _Transition(nn.Module):
    def __init__(self):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class PreResNet(nn.Module):
    def __init__(self, args, block_config):

        # Network-level hyperparameters
        self.block_config = block_config
        self.dataset = args['dataset']
        self.n_classes = args['n_classes']

        # Layer-level hyperparameters
        self.args = args

        super(PreResNet, self).__init__()

        if self.dataset in ['cifar10', 'cifar100']:
            i_channels = 16
            i_features = [
                ('conv0', nn.Conv2d(3, i_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ]
            last_pool = 8
        elif self.dataset == 'fmnist':
            i_channels = 16
            i_features = [
                ('conv0', nn.Conv2d(1, i_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ]
            last_pool = 7
        elif self.dataset == 'tinyimg':
            i_channels = 16
            i_features = [
                ('conv0', nn.Conv2d(3, i_channels, kernel_size=3, stride=2, padding=1, bias=False)),
            ]
            last_pool = 8
        elif self.dataset == 'imagenet':
            i_channels = 64
            i_features = [
                ('conv0', nn.Conv2d(3, i_channels, kernel_size=7, stride=2, padding=3, bias=False)),
                ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]
            last_pool = 7
        else:
            raise NotImplementedError()
        self.features = nn.Sequential(OrderedDict(i_features))

        planes = i_channels
        for i, n_layers in enumerate(self.block_config):
            stage = _ResidualBlock(args=args, n_layers=n_layers,
                                   in_channels=i_channels, planes=planes)
            self.features.add_module('block%d' % (i + 1), stage)
            if i != len(self.block_config) - 1:
                self.features.add_module('trans%d' % (i + 1), _Transition())
                i_channels = planes * PreBottleneck.expansion
                planes = planes * 2

        out_channels = planes * PreBottleneck.expansion
        self.features.add_module('norm_last', nn.BatchNorm2d(out_channels))
        self.features.add_module('relu_last', nn.ReLU(inplace=True))
        self.features.add_module('pool_last', nn.AvgPool2d(last_pool))

        self.classifier = nn.Linear(out_channels, self.n_classes)

        self.reset()

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


def resnet164(hparams):
    return PreResNet(hparams, block_config=[18, 18, 18])

def resnet_sconv164(hparams):
    hparams['use_sconv'] = True
    return PreResNet(hparams, block_config=[18, 18, 18])