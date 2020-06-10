"""
Objects as Points
https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/resnet_dcn.py

Simple Baselines for Human Pose Estimationand Tracking
https://github.com/microsoft/human-pose-estimation.pytorch/blob/master/lib/models/pose_resnet.py
"""

import math
from torch import nn
from ..basenet import create_basenet
from ..basenet.deform_conv import DeformConv2d



BN_MOMENTUM = 0.1


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class PoseResNet(nn.Module):
    def __init__(self, basenet, num_deconv_layers=3, use_dcn=True, frozen_bn=False):
        super().__init__()
        self.deconv_with_bias = False
        basenet, _, _ = create_basenet(basenet, pretrained='imagenet', frozen_batchnorm=frozen_bn)
        self.basenet = nn.Sequential(*basenet)
        self.out_channels = basenet[-1].out_channels

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            num_deconv_layers,
            [256, 128, 64],
            [4, 4, 4],
            use_dcn
        )

        # init deconv weights from normal distribution
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels, use_dcn):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            if use_dcn:
                fc = DeformConv2d(self.out_channels, planes, kernel_size=3, stride=1, padding=1)
            else:
                fc = nn.Conv2d(self.out_channels, planes,
                               kernel_size=3, stride=1,
                               padding=1, dilation=1, bias=False)
                fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                in_channels=planes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.out_channels = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        return deconv_kernel, padding, output_padding

    def forward(self, x):
        x = self.basenet(x)
        x = self.deconv_layers(x)
        return x
