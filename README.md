# Introduction

This directory contains the code for the paper:

**Training CNNs with Selective Allocation of Channels (ICML 2019)**.

# Requirements

- `python3`
- `torch >= 0.4.0`
- `torchvision`
- `numpy`
- `tensorboardX`

# How to run

```shell
### Train the baseline DenseNet-40 model
$ CUDA_VISIBLE_DEVICES=0 python main.py experiments/cifar10_densenet40.json

### Train DenseNet-40 with channel-selectivity (DenseNet-SConv-40)
$ CUDA_VISIBLE_DEVICES=1 python main.py experiments/cifar10_densenet_sconv40.json

### In case `tensorboard` is installed, you can also track the current training progress 
$ tensorboard --logdir=./logs
```
