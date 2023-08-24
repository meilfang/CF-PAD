import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import timm
from functools import partial
from torch import Tensor

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def norm_feature(feature, p=2, dim=1):
    feature_norm = torch.norm(feature, p=p, dim=dim, keepdim=True).clamp(min=1e-12) ** 0.5 * (2) ** 0.5
    feature = torch.div(feature, feature_norm)
    return feature


class ChannelShuffleCustom(nn.Module):
    def __init__(self, groups=16):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        if not self.training:
            return x

        batch, channels, height, width = x.size()
        assert (channels % self.groups == 0)
        channels_per_group = channels // self.groups

        x = x.view(batch, channels_per_group, self.groups, height, width)
        x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(batch, channels, height, width)
        return x


class RandomZero(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x

        mask = torch.ones_like(x,  dtype=x.dtype, device=x.device)
        channel_size = x.size(1)
        zero_index = int(self.p * channel_size)
        perm = torch.randperm(channel_size-1)[:zero_index]
        mask[:, perm, :, :] = 1e-8
        return x * mask


class RandomReplace(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training:
            return x

        channel_size = x.size(1)
        replaced_index = int(self.p * channel_size)
        perm = torch.randperm(channel_size-1)[:replaced_index]
        x[:, perm, :, :] = x[:, perm+1, :, :].clone()

        return x
