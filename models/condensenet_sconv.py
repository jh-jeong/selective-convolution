from collections import OrderedDict
import math

import torch
import torch.nn as nn

from selective_convolution import SelectiveConv2d
from models import BRC


class _CondenseSConvLayer(nn.Module):
    def __init__(self, n_channels, growth_rate, args):
        super(_CondenseSConvLayer, self).__init__()
        gamma = args['gamma']
        dropout_rate = args['dropout_rate']

        self.brc_1 = SelectiveConv2d(n_channels, 4*growth_rate, dropout_rate=dropout_rate,
                                     gamma=gamma, K=0, N_max=None)
        self.brc_2 = BRC(4*growth_rate, growth_rate,
                         kernel_size=3, padding=1, groups=4)

    def forward(self, x):
        x_ = self.brc_1(x)
        x_ = self.brc_2(x_)
        return torch.cat([x, x_], 1)


class _CondenseSConvBlock(nn.Sequential):
    def __init__(self, args, n_layers, n_channels, growth_rate):
        super(_CondenseSConvBlock, self).__init__()
        for i in range(n_layers):
            layer = _CondenseSConvLayer(n_channels + i*growth_rate, growth_rate, args)
            self.add_module('layer%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class CondenseNetSConv(nn.Module):
    def __init__(self, args, block_config, growth_rates):

        # Network-level hyperparameters
        self.block_config = block_config
        self.growth_rates = growth_rates
        self.dataset = args['dataset']
        self.n_classes = args['n_classes']

        # Layer-level hyperparameters
        self.args = args

        super(CondenseNetSConv, self).__init__()

        i_channels = 2 * self.growth_rates[0]
        self.conv0 = nn.Conv2d(3, i_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.features = nn.Sequential()

        n_channels = i_channels
        for i, n_layers in enumerate(self.block_config):
            growth_rate = self.growth_rates[i]
            block = _CondenseSConvBlock(args=args, n_layers=n_layers,
                                        n_channels=n_channels, growth_rate=growth_rate)
            self.features.add_module('block%d' % (i + 1), block)
            n_channels = n_channels + n_layers * growth_rate
            if i != len(self.block_config) - 1:
                trans = _Transition()
                self.features.add_module('trans%d' % (i + 1), trans)

        self.features.add_module('norm_last', nn.BatchNorm2d(n_channels))
        self.features.add_module('relu_last', nn.ReLU(inplace=True))
        self.features.add_module('pool_last', nn.AvgPool2d(8))

        self.classifier = nn.Linear(n_channels, self.n_classes)

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
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        y = self.conv0(x)
        features = self.features(y)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


def condensenet_sconv182(hparams):
    return CondenseNetSConv(hparams, block_config=[30, 30, 30], growth_rates=[12, 24, 48])
