"""Tensorflow like
ref: https://github.com/rwightman/pytorch-image-models
"""
from typing import List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import tup_pair

# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


# Dynamically pad input x with 'SAME' padding for conv with specified args
def pad_same(x, k, s, d=1):
    ih, iw = x.size()[-2:]
    k = tup_pair(k)
    s = tup_pair(s)
    d = tup_pair(d)
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return x


def conv2d_same(x, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, stride: Tuple[int, int] = (1, 1), dilation: Tuple[int, int] = (1, 1), groups: int = 1):
    x = pad_same(x, weight.shape[-2:], stride, dilation)
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """ Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        return conv2d_same(x, self.weight, self.bias, self.stride, self.dilation, self.groups)


def avg_pool2d_same(x, kernel_size: List[int], stride: List[int], ceil_mode: bool = False, count_include_pad: bool = True):
    x = pad_same(x, kernel_size, stride)
    return F.avg_pool2d(x, kernel_size, stride, (0, 0), ceil_mode, count_include_pad)


class AvgPool2dSame(nn.AvgPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """
    def __init__(self, kernel_size, stride=None, ceil_mode=False, count_include_pad=True):
        super(AvgPool2dSame, self).__init__(kernel_size, stride, 0, ceil_mode, count_include_pad)

    def forward(self, x):
        return avg_pool2d_same(
            x, self.kernel_size, self.stride, self.ceil_mode, self.count_include_pad)


def max_pool2d_same(x, kernel_size, stride=None, dilation=1, ceil_mode=False, return_indices=False):
    x = pad_same(x, kernel_size, stride)
    return F.max_pool2d(x, kernel_size, stride, (0, 0), dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)


class MaxPool2dSame(nn.MaxPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """
    def __init__(self, kernel_size, stride=None, dilation=1, return_indices=False, ceil_mode=False):
        super(MaxPool2dSame, self).__init__(kernel_size, stride, 0, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)

    def forward(self, x):
        return max_pool2d_same(x, self.kernel_size, self.stride, dilation=self.dilation, return_indices=self.return_indices, ceil_mode=self.ceil_mode)