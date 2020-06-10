import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import Decoder, get_num_of_channels


class LateralBlock(nn.Module):
    """
    Feature Pyramid Networks for Object Detection
    https://arxiv.org/abs/1612.03144
    """
    def __init__(self, c_planes, p_planes, out_planes):
        super().__init__()
        self.lateral = nn.Conv2d(c_planes,  p_planes,   kernel_size=1, padding=0, stride=1)
        self.top = nn.Conv2d(p_planes,  out_planes, kernel_size=3, padding=1, stride=1)

    def forward(self, c, p):
        c = self.lateral(c)
        size = c.size()[-2:]
        p = F.interpolate(p, size=size, mode='nearest')
        p = p + c
        p = self.top(p)

        return p

    @property
    def out_channels(self):
        return self.top.out_channels


def lateral_block(out_planes):
    return lambda c_layer, p_layer: LateralBlock(get_num_of_channels(c_layer),
                                                 get_num_of_channels(p_layer),
                                                 out_planes)


class LateralDecoder(nn.Module):
    """mix UNet and FPN"""
    def __init__(self, c_planes, p_planes, out_planes):
        super().__init__()
        self.out_channels = c_planes + out_planes
        self.decoder = Decoder(in_channels=p_planes, middle_channels=0, out_channels=out_planes)

    def forward(self, c , p):
        _,_, H, W = c.size()
        p = self.decoder(p)
        p = p[:, :, :H, :W]
        return torch.cat((c, p), dim=1)


def lateral_decoder(out_planes):
    return lambda c_layer, p_layer: LateralDecoder(get_num_of_channels(c_layer),
                                                   get_num_of_channels(p_layer),
                                                   out_planes)


def conv_relu(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                         nn.ReLU(inplace=True))


def deconv_relu(in_channels, out_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                         nn.ReLU(inplace=True))


class ThreeWayBlock(nn.Module):
    '''
    Residual Features and Unified Prediction Network for Single Stage Detection
    https://arxiv.org/pdf/1707.05031.pdf
    '''
    def __init__(self, c_channels, p_channels, out_channels):
        super().__init__()

        self.branch1 = conv_relu(c_channels, out_channels, 1)

        self.branch2 = nn.Sequential(
            conv_relu(c_channels, out_channels // 2, 1),
            conv_relu(out_channels // 2, out_channels // 2, 3, padding=1),
            conv_relu(out_channels // 2, out_channels, 1))
        
        self.branch3 = nn.Sequential(
            conv_relu(p_channels, out_channels // 2, 3, padding=1),
            deconv_relu(out_channels // 2, out_channels // 2),
            conv_relu(out_channels // 2, out_channels, 1))

    def forward(self, c, p):
        _,_,H,W = c.size()
        return self.branch1(c) + self.branch2(c) + self.branch3(p)[:,:,:H,:W]

    @property
    def out_channels(self):
        return self.branch1[0].out_channels