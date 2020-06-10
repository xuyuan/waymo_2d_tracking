# model converted from https://github.com/Microsoft/MMdnn/tree/master/mmdnn/conversion/darknet/README.md

import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_normalization(dim, name, **kwargs):
    if dim == 1:
        layer = nn.BatchNorm1d(**kwargs)
    elif dim == 2:
        layer = nn.BatchNorm2d(**kwargs)
    elif dim == 3:
        layer = nn.BatchNorm3d(**kwargs)
    else:
        raise NotImplementedError()

    return layer


def conv(dim, name, **kwargs):
    if dim == 1:
        layer = nn.Conv1d(**kwargs)
    elif dim == 2:
        layer = nn.Conv2d(**kwargs)
    elif dim == 3:
        layer = nn.Conv3d(**kwargs)
    else:
        raise NotImplementedError()

    return layer


class Model0(nn.Module):
    def __init__(self, model1, model2):
        super(Model0, self).__init__()
        self.model1 = model1
        self.model2 = model2

        self.layer1_conv = conv(2, name='layer1-conv', in_channels=3, out_channels=32, kernel_size=(3, 3),
                                stride=(1, 1), groups=1, bias=False)
        self.layer1_bn = batch_normalization(2, 'layer1-bn', num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.layer2_conv = conv(2, name='layer2-conv', in_channels=32, out_channels=64, kernel_size=(3, 3),
                                stride=(2, 2), groups=1, bias=False)
        self.layer2_bn = batch_normalization(2, 'layer2-bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.layer3_conv = conv(2, name='layer3-conv', in_channels=64, out_channels=32, kernel_size=(1, 1),
                                stride=(1, 1), groups=1, bias=False)
        self.layer3_bn = batch_normalization(2, 'layer3-bn', num_features=32, eps=9.999999747378752e-06, momentum=0.0)
        self.layer4_conv = conv(2, name='layer4-conv', in_channels=32, out_channels=64, kernel_size=(3, 3),
                                stride=(1, 1), groups=1, bias=False)
        self.layer4_bn = batch_normalization(2, 'layer4-bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.layer6_conv = conv(2, name='layer6-conv', in_channels=64, out_channels=128, kernel_size=(3, 3),
                                stride=(2, 2), groups=1, bias=False)
        self.layer6_bn = batch_normalization(2, 'layer6-bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.layer7_conv = conv(2, name='layer7-conv', in_channels=128, out_channels=64, kernel_size=(1, 1),
                                stride=(1, 1), groups=1, bias=False)
        self.layer7_bn = batch_normalization(2, 'layer7-bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.layer8_conv = conv(2, name='layer8-conv', in_channels=64, out_channels=128, kernel_size=(3, 3),
                                stride=(1, 1), groups=1, bias=False)
        self.layer8_bn = batch_normalization(2, 'layer8-bn', num_features=128, eps=9.999999747378752e-06, momentum=0.0)
        self.layer10_conv = conv(2, name='layer10-conv', in_channels=128, out_channels=64, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer10_bn = batch_normalization(2, 'layer10-bn', num_features=64, eps=9.999999747378752e-06, momentum=0.0)
        self.layer11_conv = conv(2, name='layer11-conv', in_channels=64, out_channels=128, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer11_bn = batch_normalization(2, 'layer11-bn', num_features=128, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer13_conv = conv(2, name='layer13-conv', in_channels=128, out_channels=256, kernel_size=(3, 3),
                                 stride=(2, 2), groups=1, bias=False)
        self.layer13_bn = batch_normalization(2, 'layer13-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer14_conv = conv(2, name='layer14-conv', in_channels=256, out_channels=128, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer14_bn = batch_normalization(2, 'layer14-bn', num_features=128, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer15_conv = conv(2, name='layer15-conv', in_channels=128, out_channels=256, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer15_bn = batch_normalization(2, 'layer15-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer17_conv = conv(2, name='layer17-conv', in_channels=256, out_channels=128, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer17_bn = batch_normalization(2, 'layer17-bn', num_features=128, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer18_conv = conv(2, name='layer18-conv', in_channels=128, out_channels=256, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer18_bn = batch_normalization(2, 'layer18-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer20_conv = conv(2, name='layer20-conv', in_channels=256, out_channels=128, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer20_bn = batch_normalization(2, 'layer20-bn', num_features=128, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer21_conv = conv(2, name='layer21-conv', in_channels=128, out_channels=256, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer21_bn = batch_normalization(2, 'layer21-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer23_conv = conv(2, name='layer23-conv', in_channels=256, out_channels=128, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer23_bn = batch_normalization(2, 'layer23-bn', num_features=128, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer24_conv = conv(2, name='layer24-conv', in_channels=128, out_channels=256, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer24_bn = batch_normalization(2, 'layer24-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer26_conv = conv(2, name='layer26-conv', in_channels=256, out_channels=128, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer26_bn = batch_normalization(2, 'layer26-bn', num_features=128, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer27_conv = conv(2, name='layer27-conv', in_channels=128, out_channels=256, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer27_bn = batch_normalization(2, 'layer27-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer29_conv = conv(2, name='layer29-conv', in_channels=256, out_channels=128, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer29_bn = batch_normalization(2, 'layer29-bn', num_features=128, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer30_conv = conv(2, name='layer30-conv', in_channels=128, out_channels=256, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer30_bn = batch_normalization(2, 'layer30-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer32_conv = conv(2, name='layer32-conv', in_channels=256, out_channels=128, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer32_bn = batch_normalization(2, 'layer32-bn', num_features=128, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer33_conv = conv(2, name='layer33-conv', in_channels=128, out_channels=256, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer33_bn = batch_normalization(2, 'layer33-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer35_conv = conv(2, name='layer35-conv', in_channels=256, out_channels=128, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer35_bn = batch_normalization(2, 'layer35-bn', num_features=128, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer36_conv = conv(2, name='layer36-conv', in_channels=128, out_channels=256, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer36_bn = batch_normalization(2, 'layer36-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer38_conv = conv(2, name='layer38-conv', in_channels=256, out_channels=512, kernel_size=(3, 3),
                                 stride=(2, 2), groups=1, bias=False)
        self.layer38_bn = batch_normalization(2, 'layer38-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer39_conv = conv(2, name='layer39-conv', in_channels=512, out_channels=256, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer39_bn = batch_normalization(2, 'layer39-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer40_conv = conv(2, name='layer40-conv', in_channels=256, out_channels=512, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer40_bn = batch_normalization(2, 'layer40-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer42_conv = conv(2, name='layer42-conv', in_channels=512, out_channels=256, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer42_bn = batch_normalization(2, 'layer42-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer43_conv = conv(2, name='layer43-conv', in_channels=256, out_channels=512, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer43_bn = batch_normalization(2, 'layer43-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer45_conv = conv(2, name='layer45-conv', in_channels=512, out_channels=256, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer45_bn = batch_normalization(2, 'layer45-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer46_conv = conv(2, name='layer46-conv', in_channels=256, out_channels=512, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer46_bn = batch_normalization(2, 'layer46-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer48_conv = conv(2, name='layer48-conv', in_channels=512, out_channels=256, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer48_bn = batch_normalization(2, 'layer48-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer49_conv = conv(2, name='layer49-conv', in_channels=256, out_channels=512, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer49_bn = batch_normalization(2, 'layer49-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer51_conv = conv(2, name='layer51-conv', in_channels=512, out_channels=256, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer51_bn = batch_normalization(2, 'layer51-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer52_conv = conv(2, name='layer52-conv', in_channels=256, out_channels=512, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer52_bn = batch_normalization(2, 'layer52-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer54_conv = conv(2, name='layer54-conv', in_channels=512, out_channels=256, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer54_bn = batch_normalization(2, 'layer54-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer55_conv = conv(2, name='layer55-conv', in_channels=256, out_channels=512, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer55_bn = batch_normalization(2, 'layer55-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer57_conv = conv(2, name='layer57-conv', in_channels=512, out_channels=256, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer57_bn = batch_normalization(2, 'layer57-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer58_conv = conv(2, name='layer58-conv', in_channels=256, out_channels=512, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer58_bn = batch_normalization(2, 'layer58-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer60_conv = conv(2, name='layer60-conv', in_channels=512, out_channels=256, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer60_bn = batch_normalization(2, 'layer60-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer61_conv = conv(2, name='layer61-conv', in_channels=256, out_channels=512, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer61_bn = batch_normalization(2, 'layer61-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer63_conv = conv(2, name='layer63-conv', in_channels=512, out_channels=1024, kernel_size=(3, 3),
                                 stride=(2, 2), groups=1, bias=False)
        self.layer63_bn = batch_normalization(2, 'layer63-bn', num_features=1024, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer64_conv = conv(2, name='layer64-conv', in_channels=1024, out_channels=512, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer64_bn = batch_normalization(2, 'layer64-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer65_conv = conv(2, name='layer65-conv', in_channels=512, out_channels=1024, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer65_bn = batch_normalization(2, 'layer65-bn', num_features=1024, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer67_conv = conv(2, name='layer67-conv', in_channels=1024, out_channels=512, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer67_bn = batch_normalization(2, 'layer67-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer68_conv = conv(2, name='layer68-conv', in_channels=512, out_channels=1024, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer68_bn = batch_normalization(2, 'layer68-bn', num_features=1024, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer70_conv = conv(2, name='layer70-conv', in_channels=1024, out_channels=512, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer70_bn = batch_normalization(2, 'layer70-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer71_conv = conv(2, name='layer71-conv', in_channels=512, out_channels=1024, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer71_bn = batch_normalization(2, 'layer71-bn', num_features=1024, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer73_conv = conv(2, name='layer73-conv', in_channels=1024, out_channels=512, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer73_bn = batch_normalization(2, 'layer73-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer74_conv = conv(2, name='layer74-conv', in_channels=512, out_channels=1024, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer74_bn = batch_normalization(2, 'layer74-bn', num_features=1024, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer76_conv = conv(2, name='layer76-conv', in_channels=1024, out_channels=512, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer76_bn = batch_normalization(2, 'layer76-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer77_conv = conv(2, name='layer77-conv', in_channels=512, out_channels=1024, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer77_bn = batch_normalization(2, 'layer77-bn', num_features=1024, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer78_conv = conv(2, name='layer78-conv', in_channels=1024, out_channels=512, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer78_bn = batch_normalization(2, 'layer78-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer79_conv = conv(2, name='layer79-conv', in_channels=512, out_channels=1024, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer79_bn = batch_normalization(2, 'layer79-bn', num_features=1024, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer80_conv = conv(2, name='layer80-conv', in_channels=1024, out_channels=512, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer80_bn = batch_normalization(2, 'layer80-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer81_conv = conv(2, name='layer81-conv', in_channels=512, out_channels=1024, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer85_conv = conv(2, name='layer85-conv', in_channels=512, out_channels=256, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer81_bn = batch_normalization(2, 'layer81-bn', num_features=1024, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer85_bn = batch_normalization(2, 'layer85-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer82_conv = conv(2, name='layer82-conv', in_channels=1024, out_channels=255, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=True)

        self.layer88_conv = conv(2, name='layer88-conv', in_channels=768, out_channels=256, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer88_bn = batch_normalization(2, 'layer88-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer89_conv = conv(2, name='layer89-conv', in_channels=256, out_channels=512, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer89_bn = batch_normalization(2, 'layer89-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer90_conv = conv(2, name='layer90-conv', in_channels=512, out_channels=256, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer90_bn = batch_normalization(2, 'layer90-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer91_conv = conv(2, name='layer91-conv', in_channels=256, out_channels=512, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer91_bn = batch_normalization(2, 'layer91-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer92_conv = conv(2, name='layer92-conv', in_channels=512, out_channels=256, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer92_bn = batch_normalization(2, 'layer92-bn', num_features=256, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer93_conv = conv(2, name='layer93-conv', in_channels=256, out_channels=512, kernel_size=(3, 3),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer97_conv = conv(2, name='layer97-conv', in_channels=256, out_channels=128, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=False)
        self.layer93_bn = batch_normalization(2, 'layer93-bn', num_features=512, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer97_bn = batch_normalization(2, 'layer97-bn', num_features=128, eps=9.999999747378752e-06,
                                              momentum=0.0)
        self.layer94_conv = conv(2, name='layer94-conv', in_channels=512, out_channels=255, kernel_size=(1, 1),
                                 stride=(1, 1), groups=1, bias=True)

        self.layer100_conv = conv(2, name='layer100-conv', in_channels=384, out_channels=128, kernel_size=(1, 1),
                                  stride=(1, 1), groups=1, bias=False)
        self.layer100_bn = batch_normalization(2, 'layer100-bn', num_features=128, eps=9.999999747378752e-06,
                                               momentum=0.0)
        self.layer101_conv = conv(2, name='layer101-conv', in_channels=128, out_channels=256, kernel_size=(3, 3),
                                  stride=(1, 1), groups=1, bias=False)
        self.layer101_bn = batch_normalization(2, 'layer101-bn', num_features=256, eps=9.999999747378752e-06,
                                               momentum=0.0)
        self.layer102_conv = conv(2, name='layer102-conv', in_channels=256, out_channels=128, kernel_size=(1, 1),
                                  stride=(1, 1), groups=1, bias=False)
        self.layer102_bn = batch_normalization(2, 'layer102-bn', num_features=128, eps=9.999999747378752e-06,
                                               momentum=0.0)
        self.layer103_conv = conv(2, name='layer103-conv', in_channels=128, out_channels=256, kernel_size=(3, 3),
                                  stride=(1, 1), groups=1, bias=False)
        self.layer103_bn = batch_normalization(2, 'layer103-bn', num_features=256, eps=9.999999747378752e-06,
                                               momentum=0.0)
        self.layer104_conv = conv(2, name='layer104-conv', in_channels=256, out_channels=128, kernel_size=(1, 1),
                                  stride=(1, 1), groups=1, bias=False)
        self.layer104_bn = batch_normalization(2, 'layer104-bn', num_features=128, eps=9.999999747378752e-06,
                                               momentum=0.0)
        self.layer105_conv = conv(2, name='layer105-conv', in_channels=128, out_channels=256, kernel_size=(3, 3),
                                  stride=(1, 1), groups=1, bias=False)
        self.layer105_bn = batch_normalization(2, 'layer105-bn', num_features=256, eps=9.999999747378752e-06,
                                               momentum=0.0)
        self.layer106_conv = conv(2, name='layer106-conv', in_channels=256, out_channels=255, kernel_size=(1, 1),
                                  stride=(1, 1), groups=1, bias=True)

    @property
    def out_channels(self):
        return self.layer106_conv.out_channels

    def forward(self, x):
        layer1_conv_pad = F.pad(x, (1, 1, 1, 1))
        layer1_conv = self.layer1_conv(layer1_conv_pad)
        layer1_bn = self.layer1_bn(layer1_conv)
        layer1_act = F.leaky_relu(layer1_bn, negative_slope=0.10000000149011612)
        layer2_conv_pad = F.pad(layer1_act, (1, 1, 1, 1))
        layer2_conv = self.layer2_conv(layer2_conv_pad)
        layer2_bn = self.layer2_bn(layer2_conv)
        layer2_act = F.leaky_relu(layer2_bn, negative_slope=0.10000000149011612)
        layer3_conv = self.layer3_conv(layer2_act)
        layer3_bn = self.layer3_bn(layer3_conv)
        layer3_act = F.leaky_relu(layer3_bn, negative_slope=0.10000000149011612)
        layer4_conv_pad = F.pad(layer3_act, (1, 1, 1, 1))
        layer4_conv = self.layer4_conv(layer4_conv_pad)
        layer4_bn = self.layer4_bn(layer4_conv)
        layer4_act = F.leaky_relu(layer4_bn, negative_slope=0.10000000149011612)
        layer5_shortcut = layer2_act + layer4_act
        layer6_conv_pad = F.pad(layer5_shortcut, (1, 1, 1, 1))
        layer6_conv = self.layer6_conv(layer6_conv_pad)
        layer6_bn = self.layer6_bn(layer6_conv)
        layer6_act = F.leaky_relu(layer6_bn, negative_slope=0.10000000149011612)
        layer7_conv = self.layer7_conv(layer6_act)
        layer7_bn = self.layer7_bn(layer7_conv)
        layer7_act = F.leaky_relu(layer7_bn, negative_slope=0.10000000149011612)
        layer8_conv_pad = F.pad(layer7_act, (1, 1, 1, 1))
        layer8_conv = self.layer8_conv(layer8_conv_pad)
        layer8_bn = self.layer8_bn(layer8_conv)
        layer8_act = F.leaky_relu(layer8_bn, negative_slope=0.10000000149011612)
        layer9_shortcut = layer6_act + layer8_act
        layer10_conv = self.layer10_conv(layer9_shortcut)
        layer10_bn = self.layer10_bn(layer10_conv)
        layer10_act = F.leaky_relu(layer10_bn, negative_slope=0.10000000149011612)
        layer11_conv_pad = F.pad(layer10_act, (1, 1, 1, 1))
        layer11_conv = self.layer11_conv(layer11_conv_pad)
        layer11_bn = self.layer11_bn(layer11_conv)
        layer11_act = F.leaky_relu(layer11_bn, negative_slope=0.10000000149011612)
        layer12_shortcut = layer9_shortcut + layer11_act
        layer13_conv_pad = F.pad(layer12_shortcut, (1, 1, 1, 1))
        layer13_conv = self.layer13_conv(layer13_conv_pad)
        layer13_bn = self.layer13_bn(layer13_conv)
        layer13_act = F.leaky_relu(layer13_bn, negative_slope=0.10000000149011612)
        layer14_conv = self.layer14_conv(layer13_act)
        layer14_bn = self.layer14_bn(layer14_conv)
        layer14_act = F.leaky_relu(layer14_bn, negative_slope=0.10000000149011612)
        layer15_conv_pad = F.pad(layer14_act, (1, 1, 1, 1))
        layer15_conv = self.layer15_conv(layer15_conv_pad)
        layer15_bn = self.layer15_bn(layer15_conv)
        layer15_act = F.leaky_relu(layer15_bn, negative_slope=0.10000000149011612)
        layer16_shortcut = layer13_act + layer15_act
        layer17_conv = self.layer17_conv(layer16_shortcut)
        layer17_bn = self.layer17_bn(layer17_conv)
        layer17_act = F.leaky_relu(layer17_bn, negative_slope=0.10000000149011612)
        layer18_conv_pad = F.pad(layer17_act, (1, 1, 1, 1))
        layer18_conv = self.layer18_conv(layer18_conv_pad)
        layer18_bn = self.layer18_bn(layer18_conv)
        layer18_act = F.leaky_relu(layer18_bn, negative_slope=0.10000000149011612)
        layer19_shortcut = layer16_shortcut + layer18_act
        layer20_conv = self.layer20_conv(layer19_shortcut)
        layer20_bn = self.layer20_bn(layer20_conv)
        layer20_act = F.leaky_relu(layer20_bn, negative_slope=0.10000000149011612)
        layer21_conv_pad = F.pad(layer20_act, (1, 1, 1, 1))
        layer21_conv = self.layer21_conv(layer21_conv_pad)
        layer21_bn = self.layer21_bn(layer21_conv)
        layer21_act = F.leaky_relu(layer21_bn, negative_slope=0.10000000149011612)
        layer22_shortcut = layer19_shortcut + layer21_act
        layer23_conv = self.layer23_conv(layer22_shortcut)
        layer23_bn = self.layer23_bn(layer23_conv)
        layer23_act = F.leaky_relu(layer23_bn, negative_slope=0.10000000149011612)
        layer24_conv_pad = F.pad(layer23_act, (1, 1, 1, 1))
        layer24_conv = self.layer24_conv(layer24_conv_pad)
        layer24_bn = self.layer24_bn(layer24_conv)
        layer24_act = F.leaky_relu(layer24_bn, negative_slope=0.10000000149011612)
        layer25_shortcut = layer22_shortcut + layer24_act
        layer26_conv = self.layer26_conv(layer25_shortcut)
        layer26_bn = self.layer26_bn(layer26_conv)
        layer26_act = F.leaky_relu(layer26_bn, negative_slope=0.10000000149011612)
        layer27_conv_pad = F.pad(layer26_act, (1, 1, 1, 1))
        layer27_conv = self.layer27_conv(layer27_conv_pad)
        layer27_bn = self.layer27_bn(layer27_conv)
        layer27_act = F.leaky_relu(layer27_bn, negative_slope=0.10000000149011612)
        layer28_shortcut = layer25_shortcut + layer27_act
        layer29_conv = self.layer29_conv(layer28_shortcut)
        layer29_bn = self.layer29_bn(layer29_conv)
        layer29_act = F.leaky_relu(layer29_bn, negative_slope=0.10000000149011612)
        layer30_conv_pad = F.pad(layer29_act, (1, 1, 1, 1))
        layer30_conv = self.layer30_conv(layer30_conv_pad)
        layer30_bn = self.layer30_bn(layer30_conv)
        layer30_act = F.leaky_relu(layer30_bn, negative_slope=0.10000000149011612)
        layer31_shortcut = layer28_shortcut + layer30_act
        layer32_conv = self.layer32_conv(layer31_shortcut)
        layer32_bn = self.layer32_bn(layer32_conv)
        layer32_act = F.leaky_relu(layer32_bn, negative_slope=0.10000000149011612)
        layer33_conv_pad = F.pad(layer32_act, (1, 1, 1, 1))
        layer33_conv = self.layer33_conv(layer33_conv_pad)
        layer33_bn = self.layer33_bn(layer33_conv)
        layer33_act = F.leaky_relu(layer33_bn, negative_slope=0.10000000149011612)
        layer34_shortcut = layer31_shortcut + layer33_act
        layer35_conv = self.layer35_conv(layer34_shortcut)
        layer35_bn = self.layer35_bn(layer35_conv)
        layer35_act = F.leaky_relu(layer35_bn, negative_slope=0.10000000149011612)
        layer36_conv_pad = F.pad(layer35_act, (1, 1, 1, 1))
        layer36_conv = self.layer36_conv(layer36_conv_pad)
        layer36_bn = self.layer36_bn(layer36_conv)
        layer36_act = F.leaky_relu(layer36_bn, negative_slope=0.10000000149011612)
        layer37_shortcut = layer34_shortcut + layer36_act
        layer38_conv_pad = F.pad(layer37_shortcut, (1, 1, 1, 1))
        layer38_conv = self.layer38_conv(layer38_conv_pad)
        layer38_bn = self.layer38_bn(layer38_conv)
        layer38_act = F.leaky_relu(layer38_bn, negative_slope=0.10000000149011612)
        layer39_conv = self.layer39_conv(layer38_act)
        layer39_bn = self.layer39_bn(layer39_conv)
        layer39_act = F.leaky_relu(layer39_bn, negative_slope=0.10000000149011612)
        layer40_conv_pad = F.pad(layer39_act, (1, 1, 1, 1))
        layer40_conv = self.layer40_conv(layer40_conv_pad)
        layer40_bn = self.layer40_bn(layer40_conv)
        layer40_act = F.leaky_relu(layer40_bn, negative_slope=0.10000000149011612)
        layer41_shortcut = layer38_act + layer40_act
        layer42_conv = self.layer42_conv(layer41_shortcut)
        layer42_bn = self.layer42_bn(layer42_conv)
        layer42_act = F.leaky_relu(layer42_bn, negative_slope=0.10000000149011612)
        layer43_conv_pad = F.pad(layer42_act, (1, 1, 1, 1))
        layer43_conv = self.layer43_conv(layer43_conv_pad)
        layer43_bn = self.layer43_bn(layer43_conv)
        layer43_act = F.leaky_relu(layer43_bn, negative_slope=0.10000000149011612)
        layer44_shortcut = layer41_shortcut + layer43_act
        layer45_conv = self.layer45_conv(layer44_shortcut)
        layer45_bn = self.layer45_bn(layer45_conv)
        layer45_act = F.leaky_relu(layer45_bn, negative_slope=0.10000000149011612)
        layer46_conv_pad = F.pad(layer45_act, (1, 1, 1, 1))
        layer46_conv = self.layer46_conv(layer46_conv_pad)
        layer46_bn = self.layer46_bn(layer46_conv)
        layer46_act = F.leaky_relu(layer46_bn, negative_slope=0.10000000149011612)
        layer47_shortcut = layer44_shortcut + layer46_act
        layer48_conv = self.layer48_conv(layer47_shortcut)
        layer48_bn = self.layer48_bn(layer48_conv)
        layer48_act = F.leaky_relu(layer48_bn, negative_slope=0.10000000149011612)
        layer49_conv_pad = F.pad(layer48_act, (1, 1, 1, 1))
        layer49_conv = self.layer49_conv(layer49_conv_pad)
        layer49_bn = self.layer49_bn(layer49_conv)
        layer49_act = F.leaky_relu(layer49_bn, negative_slope=0.10000000149011612)
        layer50_shortcut = layer47_shortcut + layer49_act
        layer51_conv = self.layer51_conv(layer50_shortcut)
        layer51_bn = self.layer51_bn(layer51_conv)
        layer51_act = F.leaky_relu(layer51_bn, negative_slope=0.10000000149011612)
        layer52_conv_pad = F.pad(layer51_act, (1, 1, 1, 1))
        layer52_conv = self.layer52_conv(layer52_conv_pad)
        layer52_bn = self.layer52_bn(layer52_conv)
        layer52_act = F.leaky_relu(layer52_bn, negative_slope=0.10000000149011612)
        layer53_shortcut = layer50_shortcut + layer52_act
        layer54_conv = self.layer54_conv(layer53_shortcut)
        layer54_bn = self.layer54_bn(layer54_conv)
        layer54_act = F.leaky_relu(layer54_bn, negative_slope=0.10000000149011612)
        layer55_conv_pad = F.pad(layer54_act, (1, 1, 1, 1))
        layer55_conv = self.layer55_conv(layer55_conv_pad)
        layer55_bn = self.layer55_bn(layer55_conv)
        layer55_act = F.leaky_relu(layer55_bn, negative_slope=0.10000000149011612)
        layer56_shortcut = layer53_shortcut + layer55_act
        layer57_conv = self.layer57_conv(layer56_shortcut)
        layer57_bn = self.layer57_bn(layer57_conv)
        layer57_act = F.leaky_relu(layer57_bn, negative_slope=0.10000000149011612)
        layer58_conv_pad = F.pad(layer57_act, (1, 1, 1, 1))
        layer58_conv = self.layer58_conv(layer58_conv_pad)
        layer58_bn = self.layer58_bn(layer58_conv)
        layer58_act = F.leaky_relu(layer58_bn, negative_slope=0.10000000149011612)
        layer59_shortcut = layer56_shortcut + layer58_act
        layer60_conv = self.layer60_conv(layer59_shortcut)
        layer60_bn = self.layer60_bn(layer60_conv)
        layer60_act = F.leaky_relu(layer60_bn, negative_slope=0.10000000149011612)
        layer61_conv_pad = F.pad(layer60_act, (1, 1, 1, 1))
        layer61_conv = self.layer61_conv(layer61_conv_pad)
        layer61_bn = self.layer61_bn(layer61_conv)
        layer61_act = F.leaky_relu(layer61_bn, negative_slope=0.10000000149011612)
        layer62_shortcut = layer59_shortcut + layer61_act
        layer63_conv_pad = F.pad(layer62_shortcut, (1, 1, 1, 1))
        layer63_conv = self.layer63_conv(layer63_conv_pad)
        layer63_bn = self.layer63_bn(layer63_conv)
        layer63_act = F.leaky_relu(layer63_bn, negative_slope=0.10000000149011612)
        layer64_conv = self.layer64_conv(layer63_act)
        layer64_bn = self.layer64_bn(layer64_conv)
        layer64_act = F.leaky_relu(layer64_bn, negative_slope=0.10000000149011612)
        layer65_conv_pad = F.pad(layer64_act, (1, 1, 1, 1))
        layer65_conv = self.layer65_conv(layer65_conv_pad)
        layer65_bn = self.layer65_bn(layer65_conv)
        layer65_act = F.leaky_relu(layer65_bn, negative_slope=0.10000000149011612)
        layer66_shortcut = layer63_act + layer65_act
        layer67_conv = self.layer67_conv(layer66_shortcut)
        layer67_bn = self.layer67_bn(layer67_conv)
        layer67_act = F.leaky_relu(layer67_bn, negative_slope=0.10000000149011612)
        layer68_conv_pad = F.pad(layer67_act, (1, 1, 1, 1))
        layer68_conv = self.layer68_conv(layer68_conv_pad)
        layer68_bn = self.layer68_bn(layer68_conv)
        layer68_act = F.leaky_relu(layer68_bn, negative_slope=0.10000000149011612)
        layer69_shortcut = layer66_shortcut + layer68_act
        layer70_conv = self.layer70_conv(layer69_shortcut)
        layer70_bn = self.layer70_bn(layer70_conv)
        layer70_act = F.leaky_relu(layer70_bn, negative_slope=0.10000000149011612)
        layer71_conv_pad = F.pad(layer70_act, (1, 1, 1, 1))
        layer71_conv = self.layer71_conv(layer71_conv_pad)
        layer71_bn = self.layer71_bn(layer71_conv)
        layer71_act = F.leaky_relu(layer71_bn, negative_slope=0.10000000149011612)
        layer72_shortcut = layer69_shortcut + layer71_act
        layer73_conv = self.layer73_conv(layer72_shortcut)
        layer73_bn = self.layer73_bn(layer73_conv)
        layer73_act = F.leaky_relu(layer73_bn, negative_slope=0.10000000149011612)
        layer74_conv_pad = F.pad(layer73_act, (1, 1, 1, 1))
        layer74_conv = self.layer74_conv(layer74_conv_pad)
        layer74_bn = self.layer74_bn(layer74_conv)
        layer74_act = F.leaky_relu(layer74_bn, negative_slope=0.10000000149011612)
        layer75_shortcut = layer72_shortcut + layer74_act
        layer76_conv = self.layer76_conv(layer75_shortcut)
        layer76_bn = self.layer76_bn(layer76_conv)
        layer76_act = F.leaky_relu(layer76_bn, negative_slope=0.10000000149011612)
        layer77_conv_pad = F.pad(layer76_act, (1, 1, 1, 1))
        layer77_conv = self.layer77_conv(layer77_conv_pad)
        layer77_bn = self.layer77_bn(layer77_conv)
        layer77_act = F.leaky_relu(layer77_bn, negative_slope=0.10000000149011612)
        layer78_conv = self.layer78_conv(layer77_act)
        layer78_bn = self.layer78_bn(layer78_conv)
        layer78_act = F.leaky_relu(layer78_bn, negative_slope=0.10000000149011612)
        layer79_conv_pad = F.pad(layer78_act, (1, 1, 1, 1))
        layer79_conv = self.layer79_conv(layer79_conv_pad)
        layer79_bn = self.layer79_bn(layer79_conv)
        layer79_act = F.leaky_relu(layer79_bn, negative_slope=0.10000000149011612)
        layer80_conv = self.layer80_conv(layer79_act)
        layer80_bn = self.layer80_bn(layer80_conv)
        layer80_act = F.leaky_relu(layer80_bn, negative_slope=0.10000000149011612)
        layer81_conv_pad = F.pad(layer80_act, (1, 1, 1, 1))
        layer81_conv = self.layer81_conv(layer81_conv_pad)
        layer85_conv = self.layer85_conv(layer80_act)
        layer81_bn = self.layer81_bn(layer81_conv)
        layer85_bn = self.layer85_bn(layer85_conv)
        layer81_act = F.leaky_relu(layer81_bn, negative_slope=0.10000000149011612)
        layer85_act = F.leaky_relu(layer85_bn, negative_slope=0.10000000149011612)
        layer82_conv = self.layer82_conv(layer81_act)

        layer86_upsample = F.upsample(layer85_act, scale_factor=2, mode='nearest')
        layer87_concat = torch.cat((layer86_upsample, layer62_shortcut), 1)
        layer88_conv = self.layer88_conv(layer87_concat)
        layer88_bn = self.layer88_bn(layer88_conv)
        layer88_act = F.leaky_relu(layer88_bn, negative_slope=0.10000000149011612)
        layer89_conv_pad = F.pad(layer88_act, (1, 1, 1, 1))
        layer89_conv = self.layer89_conv(layer89_conv_pad)
        layer89_bn = self.layer89_bn(layer89_conv)
        layer89_act = F.leaky_relu(layer89_bn, negative_slope=0.10000000149011612)
        layer90_conv = self.layer90_conv(layer89_act)
        layer90_bn = self.layer90_bn(layer90_conv)
        layer90_act = F.leaky_relu(layer90_bn, negative_slope=0.10000000149011612)
        layer91_conv_pad = F.pad(layer90_act, (1, 1, 1, 1))
        layer91_conv = self.layer91_conv(layer91_conv_pad)
        layer91_bn = self.layer91_bn(layer91_conv)
        layer91_act = F.leaky_relu(layer91_bn, negative_slope=0.10000000149011612)
        layer92_conv = self.layer92_conv(layer91_act)
        layer92_bn = self.layer92_bn(layer92_conv)
        layer92_act = F.leaky_relu(layer92_bn, negative_slope=0.10000000149011612)
        layer93_conv_pad = F.pad(layer92_act, (1, 1, 1, 1))
        layer93_conv = self.layer93_conv(layer93_conv_pad)
        layer97_conv = self.layer97_conv(layer92_act)
        layer93_bn = self.layer93_bn(layer93_conv)
        layer97_bn = self.layer97_bn(layer97_conv)
        layer93_act = F.leaky_relu(layer93_bn, negative_slope=0.10000000149011612)
        layer97_act = F.leaky_relu(layer97_bn, negative_slope=0.10000000149011612)
        layer94_conv = self.layer94_conv(layer93_act)

        layer98_upsample = F.upsample(layer97_act, scale_factor=2, mode='nearest')
        layer99_concat = torch.cat((layer98_upsample, layer37_shortcut), 1)
        layer100_conv = self.layer100_conv(layer99_concat)
        layer100_bn = self.layer100_bn(layer100_conv)
        layer100_act = F.leaky_relu(layer100_bn, negative_slope=0.10000000149011612)
        layer101_conv_pad = F.pad(layer100_act, (1, 1, 1, 1))
        layer101_conv = self.layer101_conv(layer101_conv_pad)
        layer101_bn = self.layer101_bn(layer101_conv)
        layer101_act = F.leaky_relu(layer101_bn, negative_slope=0.10000000149011612)
        layer102_conv = self.layer102_conv(layer101_act)
        layer102_bn = self.layer102_bn(layer102_conv)
        layer102_act = F.leaky_relu(layer102_bn, negative_slope=0.10000000149011612)
        layer103_conv_pad = F.pad(layer102_act, (1, 1, 1, 1))
        layer103_conv = self.layer103_conv(layer103_conv_pad)
        layer103_bn = self.layer103_bn(layer103_conv)
        layer103_act = F.leaky_relu(layer103_bn, negative_slope=0.10000000149011612)
        layer104_conv = self.layer104_conv(layer103_act)
        layer104_bn = self.layer104_bn(layer104_conv)
        layer104_act = F.leaky_relu(layer104_bn, negative_slope=0.10000000149011612)
        layer105_conv_pad = F.pad(layer104_act, (1, 1, 1, 1))
        layer105_conv = self.layer105_conv(layer105_conv_pad)
        layer105_bn = self.layer105_bn(layer105_conv)
        layer105_act = F.leaky_relu(layer105_bn, negative_slope=0.10000000149011612)
        layer106_conv = self.layer106_conv(layer105_act)

        # HACK
        self.model1.layer94_conv_output = layer94_conv
        self.model2.layer82_conv_output = layer82_conv

        return layer106_conv


class Model1(nn.Module):
    def forward(self, x):
        return self.layer94_conv_output


class Model2(nn.Module):
    def forward(self, x):
        return self.layer82_conv_output


class KitModel(nn.Module):

    def __init__(self):
        super(KitModel, self).__init__()
        self.model1 = Model1()
        self.model2 = Model2()
        self.model0 = Model0(self.model1, self.model2)
        self.model1.out_channels = self.model0.layer94_conv.out_channels
        self.model2.out_channels = self.model0.layer82_conv.out_channels


    def forward(self, x):
        layer106_conv = self.model0(x)
        layer94_conv = self.model1(x)
        layer82_conv = self.model2(x)
        return layer106_conv, layer94_conv, layer82_conv

    def load_state_dict(self, state_dict, strict=True):
        self.model0.load_state_dict(state_dict, strict=strict)
