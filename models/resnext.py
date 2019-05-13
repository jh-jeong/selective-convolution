from collections import OrderedDict
import math

import torch
import torch.nn as nn

from selective_convolution import SelectiveConv2d
from models import BRC


class ResNeXtBlock(nn.Module):
    def __init__(self, in_planes, out_planes, args):
        super(ResNeXtBlock, self).__init__()
        n_groups = args['n_groups']
        base_width = args['base_width']
        widen_factor = args['widen_factor']

        self.in_planes = in_planes
        self.add_planes = out_planes - in_planes
        width_ratio = out_planes // (64 * widen_factor)
        planes = base_width * width_ratio * n_groups

        if args.get('use_sconv', False):
            gamma = args['gamma']
            K = args.get('K', 3)
            N_max = args.get('N_max', None)

            self.brc_1 = SelectiveConv2d(in_planes, planes, kernel_size=1,
                                         gamma=gamma, K=K, N_max=N_max)
            self.brc_2 = BRC(planes, planes, kernel_size=3, padding=1, groups=n_groups)
            self.brc_3 = SelectiveConv2d(planes, out_planes, kernel_size=1,
                                         gamma=gamma, K=K, N_max=N_max)
        else:
            self.brc_1 = BRC(in_planes, planes, kernel_size=1)
            self.brc_2 = BRC(planes, planes, kernel_size=3, padding=1, groups=n_groups)
            self.brc_3 = BRC(planes, out_planes, kernel_size=1)

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


class _ResNeXtStage(nn.Sequential):
    def __init__(self, args, n_layers, in_planes, out_planes):
        super(_ResNeXtStage, self).__init__()

        layer = ResNeXtBlock(in_planes, out_planes, args)
        self.add_module('layer1', layer)

        for i in range(n_layers-1):
            layer = ResNeXtBlock(out_planes, out_planes, args)
            self.add_module('layer%d' % (i + 2), layer)


class _Transition(nn.Module):
    def __init__(self):
        super(_Transition, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(x)
        return x


class ResNeXt(nn.Module):
    def __init__(self, args, block_config, n_groups, base_width, widen_factor):

        # Network-level hyperparameters
        self.block_config = block_config
        self.dataset = args['dataset']
        self.n_classes = args['n_classes']

        # Layer-level hyperparameters
        args['n_groups'] = n_groups
        args['base_width'] = base_width
        args['widen_factor'] = widen_factor
        self.args = args

        super(ResNeXt, self).__init__()

        if self.dataset in ['cifar10', 'cifar100']:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
            last_pool = 8
        elif self.dataset == 'fmnist':
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)),
            ]))
            last_pool = 7
        elif self.dataset == 'tinyimg':
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)),
            ]))
            last_pool = 8
        else:
            raise NotImplementedError()

        in_channels = 64
        out_channels = in_channels * widen_factor
        for i, n_layers in enumerate(self.block_config):
            stage = _ResNeXtStage(args=args, n_layers=n_layers,
                                  in_planes=in_channels, out_planes=out_channels)
            self.features.add_module('block%d' % (i + 1), stage)
            if i != len(self.block_config) - 1:
                self.features.add_module('trans%d' % (i + 1), _Transition())
                in_channels = out_channels
                out_channels = out_channels * 2

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


def resnext29(hparams):
    return ResNeXt(hparams, block_config=[3, 3, 3], n_groups=8, base_width=64, widen_factor=4)

def resnext_sconv29(hparams):
    hparams['use_sconv'] = True
    return ResNeXt(hparams, block_config=[3, 3, 3], n_groups=8, base_width=64, widen_factor=4)