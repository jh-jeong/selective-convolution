import math

import torch
import torch.nn as nn
from torch.nn.functional import affine_grid, grid_sample
import numpy as np


def _cdf(x):
    r"""Compute the c.d.f. of the standard normal distribution."""
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def _phi(x):
    r"""Compute the p.d.f. of the standard normal distribution."""
    return (1 / math.sqrt(2 * math.pi)) * (-0.5 * x ** 2).exp()


class SelectiveConv2d(nn.Module):
    r"""Applies 2D selective convolution on an 4D input."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 dropout_rate=0., gamma=0.001, K=3, N_max=None):
        super(SelectiveConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.gamma = gamma
        self.N_max = N_max

        self._bias = None
        if K > 0:
            _ind_ptr = torch.full((in_channels,), -1, dtype=torch.long)
            _bias = torch.zeros(in_channels, 2)

            self.register_buffer('_ind_ptr', _ind_ptr)
            self._bias = nn.Parameter(_bias)

            _eye = torch.eye(2, 3)
            self.register_buffer('_eye', _eye)

        is_open = torch.ones(in_channels)
        self.register_buffer('is_open', is_open)

        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.drop = nn.Dropout(dropout_rate, inplace=False)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)

    def forward(self, input):
        x = self._distribute_channel(input)
        x = self.norm(x)
        x = self.relu(x)
        if self.dropout_rate > 0:
            x = self.drop(x)
        y = x * self.is_open.view(self.in_channels, 1, 1)
        output = self.conv(y)
        return output

    def _distribute_channel(self, input):
        if self.K == 0:
            return input

        ptr_exist = (self._ind_ptr >= 0)
        if not ptr_exist.any():
            return input

        N, _, H, W = input.size()
        loc_replace = ptr_exist.nonzero().view(-1)
        ind_load = self._ind_ptr[loc_replace]

        theta = self._eye.repeat(loc_replace.numel(), 1, 1)
        bias = self._bias[loc_replace, :]

        scale = (H + W) / 4.0
        bias = bias / scale
        theta[:, :, 2] = bias

        size = torch.Size((loc_replace.numel(), N, H, W))
        grid = affine_grid(theta, size)

        to_store = input[:, ind_load, :, :]
        to_store = to_store.permute(1, 0, 2, 3)
        to_store = grid_sample(to_store, grid, padding_mode='border')
        to_store = to_store.permute(1, 0, 2, 3)

        output = input.clone()
        output[:, loc_replace, :, :] = to_store

        return output

    def _copy_parameters(self, dst, src):
        self.conv.weight.data[:, dst, :, :] = 0

        self.norm.weight.data[dst] = self.norm.weight.data[src]
        self.norm.bias.data[dst] = self.norm.bias.data[src]
        self.norm.running_mean[dst] = self.norm.running_mean[src]
        self.norm.running_var[dst] = self.norm.running_var[src]

    @property
    def n_open(self):
        return int(self.is_open.sum())

    def open_gate(self, mask):
        assert (self.is_open[mask] == 0).all()
        self.is_open[mask] = 1

    def close_gate(self, mask):
        self.is_open[mask] = 0

    def _score(self):
        n_ecdm = self.ecdm(normalize=True)
        return n_ecdm.norm(2, 0)

    def ecdm(self, normalize=False):
        gamma = self.norm.weight.abs()
        beta = self.norm.bias
        beta_gamma = beta / (gamma + 1e-8)
        E_inp = gamma*_phi(beta_gamma) + beta*_cdf(beta_gamma)
        E_inp = E_inp * self.is_open

        weight_sum = self.conv.weight.sum(-1).sum(-1)
        ecdm = weight_sum * E_inp

        if normalize:
            ecdm = ecdm.abs()
            ecdm = ecdm / (ecdm.sum(1)+1e-8).view(-1, 1)

        return ecdm

    def realloc(self):
        if self.K == 0:
            return

        is_closed = 1 - self.is_open
        ind_free = is_closed.nonzero().view(-1)
        n_slots = ind_free.numel()
        if n_slots == 0:
            return

        score = self._score()

        if self.N_max is not None:
            z = self._ind_ptr[self._ind_ptr != -1].cpu().numpy()
            ind_alloc, ind_count = np.unique(z, return_counts=True)
            ind_over = ind_alloc[ind_count > self.N_max]
            for i in ind_over:
                score[self._ind_ptr == i.item()] = 0
                score[i.item()] = 0

        _, loc_src = score.topk(self.K)
        loc_src = loc_src[torch.randint(self.K, (n_slots,), dtype=torch.long)]

        for i, src in enumerate(loc_src):
            loc_dst = ind_free[i: i+1]
            self.open_gate(loc_dst)
            for dst in loc_dst:
                self._copy_parameters(dst, src)

            bias_src = self._bias[src, :].data

            noise = bias_src.new(1, 2).uniform_(-1.5, 1.5)
            self._bias.data[loc_dst, :] = noise

            ptr_src = self._ind_ptr[src]
            if ptr_src >= 0:
                src = ptr_src
            self._ind_ptr[loc_dst] = src

    def dealloc(self):
        n_ecdm = self.ecdm(normalize=True)
        _, ind = n_ecdm.max(0)[0].sort()

        n_close = 0
        for i in range(self.in_channels):
            vec_sum = n_ecdm[:, ind[:i+1]].sum(1).max().item()
            if vec_sum > self.gamma:
                n_close = i
                break

        if n_close > 0:
            self.close_gate(ind[:n_close])
            if self.K > 0:
                self._ind_ptr[ind[:n_close]] = -1
