from torch import nn
from torchvision import ops
from .basic import get_num_of_channels


class DeformConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        self.layer = ops.DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                      dilation=dilation, groups=groups, bias=bias)
        kernel_height, kernel_width = self.layer.kernel_size
        offset_groups = groups
        offset_channels = 2 * offset_groups * kernel_height * kernel_width
        self.offsets = nn.Conv2d(in_channels, offset_channels, kernel_size=kernel_size, padding=padding)
        self.out_channels = out_channels

    def forward(self, x):
        offset = self.offsets(x)
        return self.layer(x.float(), offset.float()).to(x)


def deform_conv2d(*args, **kwargs):
    return lambda last_layer: DeformConv2d(get_num_of_channels(last_layer), *args, **kwargs)