__all__ = []

import pkgutil
import inspect

import torch.nn as nn


class BRC(nn.Sequential):
    """Abbreviation of BN-ReLU-Conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dropout_rate=0.0, groups=1):
        super(BRC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        if dropout_rate > 0:
            self.add_module('drop', nn.Dropout(dropout_rate))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding, bias=False,
                                          groups=groups))


for loader, name, is_pkg in pkgutil.walk_packages(__path__):
    module = loader.find_module(name).load_module(name)

    for name, value in inspect.getmembers(module):
        if name.startswith('__'):
            continue

        globals()[name] = value
        if name not in __all__:
            __all__.append(name)