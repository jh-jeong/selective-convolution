# Reference: https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
from collections import OrderedDict
import math

import torch
import torch.nn as nn

from selective_convolution import SelectiveConv2d
from models import BRC


class _DenseLayer(nn.Module):
    def __init__(self, n_channels, growth_rate, args):
        super(_DenseLayer, self).__init__()
        if args.get('use_sconv', False):
            gamma = args['gamma']
            K = args.get('K', 3)
            N_max = args.get('N_max', None)
            self.brc_1 = SelectiveConv2d(n_channels, 4*growth_rate,
                                         gamma=gamma, K=K, N_max=N_max)
            self.brc_2 = SelectiveConv2d(4*growth_rate, growth_rate,
                                         kernel_size=3, padding=1,
                                         gamma=gamma, K=0, N_max=N_max)
        else:
            self.brc_1 = BRC(n_channels, 4*growth_rate, kernel_size=1)
            self.brc_2 = BRC(4*growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        x_ = self.brc_1(x)
        x_ = self.brc_2(x_)
        return torch.cat([x, x_], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, args, n_layers, n_channels, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(n_layers):
            layer = _DenseLayer(n_channels + i*growth_rate, growth_rate, args)
            self.add_module('layer%d' % (i + 1), layer)


class _Transition(nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(_Transition, self).__init__()
        if in_channels != out_channels:
            if args.get('use_sconv', False):
                gamma = args['gamma']
                K = args.get('K', 3)
                N_max = args.get('N_max', None)
                self.brc = SelectiveConv2d(in_channels, out_channels,
                                           gamma=gamma, K=K, N_max=N_max)
            else:
                self.brc = BRC(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if hasattr(self, 'brc'):
            x = self.brc(x)
        x = self.pool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, args, block_config, growth_rate=12, compression=1.0):

        # Network-level hyperparameters
        self.block_config = block_config
        self.dataset = args['dataset']
        self.n_classes = args['n_classes']
        self.growth_rate = growth_rate
        self.compression = compression

        # Layer-level hyperparameters
        self.args = args

        assert 0 < self.compression <= 1, '0 < compression <= 1'

        super(DenseNet, self).__init__()

        i_channels = 2 * self.growth_rate
        if self.dataset in ['cifar10', 'cifar100']:
            i_features = [
                ('conv0', nn.Conv2d(3, i_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ]
            last_pool = 8
        elif self.dataset == 'fmnist':
            i_features = [
                ('conv0', nn.Conv2d(1, i_channels, kernel_size=3, stride=1, padding=1, bias=False)),
            ]
            last_pool = 7
        elif self.dataset == 'tinyimg':
            i_features = [
                ('conv0', nn.Conv2d(3, i_channels, kernel_size=3, stride=2, padding=1, bias=False)),
            ]
            last_pool = 8
        elif self.dataset == 'imagenet':
            i_features = [
                ('conv0', nn.Conv2d(3, i_channels, kernel_size=7, stride=2, padding=3, bias=False)),
                ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]
            last_pool = 7
        else:
            raise NotImplementedError()
        self.features = nn.Sequential(OrderedDict(i_features))

        n_channels = i_channels
        for i, n_layers in enumerate(self.block_config):
            block = _DenseBlock(args=args, n_layers=n_layers,
                                n_channels=n_channels, growth_rate=self.growth_rate)
            self.features.add_module('block%d' % (i + 1), block)
            n_channels = n_channels + n_layers * self.growth_rate
            if i != len(self.block_config) - 1:
                trans = _Transition(args=args, in_channels=n_channels,
                                    out_channels=int(n_channels * self.compression))
                self.features.add_module('trans%d' % (i + 1), trans)
                n_channels = int(n_channels * self.compression)

        self.features.add_module('norm_last', nn.BatchNorm2d(n_channels))
        self.features.add_module('relu_last', nn.ReLU(inplace=True))
        self.features.add_module('pool_last', nn.AvgPool2d(last_pool))

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
        features = self.features(x)
        out = features.view(features.size(0), -1)
        out = self.classifier(out)
        return out


def densenet40(hparams):
    return DenseNet(hparams, growth_rate=12, block_config=[6, 6, 6], compression=1.0)

def densenet100(hparams):
    return DenseNet(hparams, growth_rate=12, block_config=[16, 16, 16], compression=1.0)

def densenet_bc190(hparams):
    return DenseNet(hparams, growth_rate=40, block_config=[31, 31, 31], compression=0.5)

def densenet_sconv40(hparams):
    hparams['use_sconv'] = True
    return DenseNet(hparams, growth_rate=12, block_config=[6, 6, 6], compression=1.0)

def densenet_sconv100(hparams):
    hparams['use_sconv'] = True
    return DenseNet(hparams, growth_rate=12, block_config=[16, 16, 16], compression=1.0)

def densenet_bc_sconv190(hparams):
    hparams['use_sconv'] = True
    return DenseNet(hparams, growth_rate=40, block_config=[31, 31, 31], compression=0.5)
