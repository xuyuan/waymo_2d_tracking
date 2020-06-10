import torch
from torch import nn
from ..basenet import get_num_of_channels, ConvBnRelu


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, up_op='deconv'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if middle_channels <= 0:
            # default
            middle_channels = out_channels * 4

        if up_op in ['deconv', 'conv_transpose']:
            # Parameters were chosen to avoid artifacts, suggested by https://distill.pub/2016/deconv-checkerboard/
            up_module = nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1)
        elif up_op in ['nearest', 'linear', 'bilinear', 'trilinear']:
            up_module = nn.Upsample(scale_factor=2, mode=up_op, align_corners=False)
            middle_channels = out_channels
        else:
            raise NotImplementedError(up_op)

        self.block = nn.Sequential(
            nn.Dropout2d(p=0.1, inplace=True),
            ConvBnRelu(in_channels, middle_channels, kernel_size=3, padding=1),
            up_module
        )

    def forward(self, *args):
        x = torch.cat(args, 1)
        return self.block(x)


def decoder(out_channels, middle_channels=0, up_op='deconv'):
    return lambda last_layer: Decoder(get_num_of_channels(last_layer), middle_channels, out_channels, up_op=up_op)
