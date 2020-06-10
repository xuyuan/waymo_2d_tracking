import torch
from ..basenet.basic import ConvBn, SeparableConv2d, Sequential, Swish
from ..basenet.tf_like import max_pool2d_same
from torch import nn
from torch.nn import functional as F


class Node(nn.Module):
    def __init__(self, in_channels, out_channels, input_offset, weighted_sum):
        super().__init__()
        self.input_offset = input_offset
        self.out_channels = out_channels

        in_channels = [in_channels[i] for i in input_offset]
        self.w = None
        if weighted_sum:
            self.w = nn.Parameter(torch.ones(len(in_channels)), requires_grad=True)

        self.resample = None
        for i in in_channels:
            if i != out_channels:
                self.resample = nn.ModuleList([self._maybe_apply_1x1(c) for c in in_channels])
                break

        self.op_after_combine = Sequential(Swish(),
                                           SeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
                                           nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3))

    def _maybe_apply_1x1(self, in_channels):
        if in_channels != self.out_channels:
            return ConvBn(in_channels=in_channels, out_channels=self.out_channels, kernel_size=1)
        return nn.Identity()

    def forward(self, inputs):
        """
        Args:
            inputs: [x0, x1, ...]
        Returns:
            x (same size as x0)
        """
        x = [inputs[i] for i in self.input_offset]
        size = x[0].shape[-2:]
        if self.resample:
            x = [m(i) for i, m in zip(x, self.resample)]
        x = [self.resize(i, size) for i in x]

        if self.w is not None:
            w = F.normalize(F.relu(self.w), p=1, dim=0, eps=0.0001)
            x = [xi * wi for xi, wi in zip(x, w)]
        x = sum(x)
        x = self.op_after_combine(x)

        inputs.append(x)
        return inputs

    def resize(self, x, size):
        if x.shape[-1] > size[-1]:
            #return F.adaptive_max_pool2d(x, size)
            stride = 2#max(x.shape[-2] // size[-2], x.shape[-1] // size[-1])
            kernel_size = stride + 1
            return max_pool2d_same(x, kernel_size=kernel_size, stride=stride)
        if x.shape[-1] < size[-1]:
            return F.interpolate(x, size=size, mode='nearest')
        return x


class BiFPNLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, weighted_sum):
        assert isinstance(in_channels, list)

        levels = len(in_channels)

        pathtd = []
        for i in range(levels - 2, -1, -1):
            offset = [i, len(in_channels) - 1]
            pathtd.append(offset)  # top down
            in_channels.append(out_channels)

        pathdt = []
        for i in range(1, levels - 1):
            pathdt.append([i, pathtd[-i][1], len(in_channels) - 1])
            in_channels.append(out_channels)
        pathdt.append([levels - 1, len(in_channels) - 1])
        inputs_offsets = pathtd + pathdt

        super().__init__(*[Node(in_channels, out_channels, i, weighted_sum) for i in inputs_offsets])
        self.levels = levels
        self.out_channels = out_channels

    def forward(self, inputs):
        inputs = inputs[-self.levels:]
        ret = super().forward(inputs)
        return ret[-self.levels:]


class BiFPN(nn.Sequential):
    def __init__(self, in_channels, out_channels, stack=1, weighted_sum=True):
        bifpn_stacks = []
        levels = len(in_channels)
        for _ in range(stack):
            bifpn_stacks.append(BiFPNLayer(in_channels=in_channels, out_channels=out_channels,
                                           weighted_sum=weighted_sum))
            in_channels = [out_channels] * levels
        super().__init__(*bifpn_stacks)
        self.out_channels = out_channels
        self.levels = levels


def bifpn(out_channels, stack=1, weighted_sum=True):
    return lambda layers: BiFPN(in_channels=[l.out_channels for l in layers], out_channels=out_channels, stack=stack,
                                weighted_sum=weighted_sum)