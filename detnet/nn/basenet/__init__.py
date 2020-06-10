from collections import OrderedDict
import warnings
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision
from .inplace_abn import ActivatedBatchNorm
from .basic import Sequential, ConvBnRelu, ConvRelu, get_num_of_channels, Swish, HardSwish, Mish, convert_activation, AdaptiveConcatPool2d, InputNormalization
from .resnet_ibn_a import Bottleneck as BottleneckIBNa


BASENET_CHOICES = ('vgg11', 'vgg13', 'vgg16', 'vgg19',
                   'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
                   #
                   'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                   'Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Resnet152',
                   'resnet18_abn', 'resnet34_abn', 'resnet50_abn', 'resnet101_abn', 'resnet152_abn',
                   'Resnet18_abn', 'Resnet34_abn', 'Resnet50_abn', 'Resnet101_abn', 'Resnet152_abn',
                   'resnet50_ibn_a', 'resnet101_ibn_a', 'resnet152_ibn_a',
                   #
                   'resnext50_32x4d', 'resnext101_32x8d', 'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d',
                   'Resnext50_32x4d', 'Resnext101_32x8d', 'Resnext101_32x16d', 'Resnext101_32x32d', 'Resnext101_32x48d',
                   'resnext101_32x4d', 'resnext101_64x4d',
                   'Resnext101_32x4d', 'Resnext101_64x4d',
                   #
                   'se_resnet50', 'se_resnet101', 'se_resnet152',
                   'Se_resnet50', 'Se_resnet101', 'Se_resnet152',
                   'se_resnet50_abn', 'se_resnet101_abn', 'se_resnet152_abn',
                   'Se_resnet50_abn', 'Se_resnet101_abn', 'Se_resnet152_abn',
                   #
                   'se_resnext50_32x4d', 'se_resnext101_32x4d', 'senet154',
                   'Se_resnext50_32x4d', 'Se_resnext101_32x4d', 'Senet154',
                   'se_resnext50_32x4d_abn', 'se_resnext101_32x4d_abn', 'senet154_abn',
                   'Se_resnext50_32x4d_abn', 'Se_resnext101_32x4d_abn', 'Senet154_abn',
                   #
                   'densenet121',
                   #
                   'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
                   'Efficientnet-b0', 'Efficientnet-b1', 'Efficientnet-b2', 'Efficientnet-b3', 'Efficientnet-b4', 'Efficientnet-b5', 'Efficientnet-b6', 'Efficientnet-b7',
                   #
                   'mobilenet_v2', 'Mobilenet_v2',
                   'mobilenet_v3',
                   #
                   'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0',
                   #
                   'squeezenet1_0', 'squeezenet1_1',
                   #
                   'darknet')


MODEL_ZOO_URL = 'https://model_zoo/'

MODEL_URLS = {
    'vgg16': {'voc': 'SSDv2_vgg16_c21-96ae2bf2.pth',
              'coco': 'SSDretina_vgg16_c81-de29503d.pth'},
    'resnet50': {#'voc':  'SSDretina_resnet50_c21-fb6036d1.pth',  # SSDretina_resnet50_c81-a584ead7.pth pretrained
                 'voc': 'SSDretina_resnet50_c21-1c85a349.pth',  # SSDretina_resnet50_c501-06095077.pth pretrained
                 'coco': 'SSDretina_resnet50_c81-a584ead7.pth',
                 'oid': 'SSDretina_resnet50_c501-06095077.pth',
                 'visdrone': 'SSDdrone_resnet50_c12-9777e250.pth',
                 'efffeu': 'SSDdrone_resnet50_c6_job2854-99e7388e.pth' # SSDretina_resnet50_c6_job2833_399f6639.pth
                },
    'resnet101': {'coco': 'SSDretina_resnet101_c81-d515d740.pth'},
    'resnext101_32x4d': {'coco': 'SSDretina_resnext101_32x4d_c81-fdb37546.pth'},
    'se_resnext50_32x4d': {'coco': 'SSDretina_se_resnext50_32x4d-c280aa00.pth'},
    'se_resnext101_32x4d': {'coco': 'SSDretina_se_resnext101_32x4d_c81-14b8f37.pth'},
    'senet154': {'coco': 'SSDretina_senet154_c81-e940bc59.pth'},
    'mobilenet_v2': {'imagenet': 'mobilenet_v2-ecbe2b56.pth',
                     'coco': 'SSDretina_mobilenet_v2_c81-4d8aaad4.pth'},
    'darknet': {'coco': 'pytorch_yolov3-60bd5e05.pth'}
}


def load_from_file_or_model_zoo(filename):
    if isinstance(filename, str) and filename.startswith('model_zoo:'):
        model_url = filename.replace('model_zoo:', MODEL_ZOO_URL)
        data = model_zoo.load_url(model_url, map_location=lambda storage, loc: storage)
    else:
        data = torch.load(filename, map_location=lambda storage, loc: storage)
    return data


def get_model_zoo_url(basenet, pretrained):
    return MODEL_ZOO_URL + MODEL_URLS[basenet][pretrained]


def vgg_base_extra(bn):
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    block = ConvBnRelu if bn else ConvRelu
    conv6 = block(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = block(1024, 1024, kernel_size=1)
    return [pool5, conv6, conv7]


def vgg(name, pretrained):
    if name == 'vgg11':
        net_class = torchvision.models.vgg11
    elif name == 'vgg13':
        net_class = torchvision.models.vgg13
    elif name == 'vgg16':
        net_class = torchvision.models.vgg16
    elif name == 'vgg19':
        net_class = torchvision.models.vgg19
    elif name == 'vgg11_bn':
        net_class = torchvision.models.vgg11_bn
    elif name == 'vgg13_bn':
        net_class = torchvision.models.vgg13_bn
    elif name == 'vgg16_bn':
        net_class = torchvision.models.vgg16_bn
    elif name == 'vgg19_bn':
        net_class = torchvision.models.vgg19_bn
    else:
        raise RuntimeError("unknown model {}".format(name))

    imagenet_pretrained = pretrained == 'imagenet'
    vgg = net_class(pretrained=imagenet_pretrained)

    # for have exact same layout as original paper
    if name == 'vgg16':
        vgg.features[16].ceil_mode = True

    bn = name.endswith('bn')
    layers = []
    l = []
    for i in range(len(vgg.features) - 1):
        if isinstance(vgg.features[i], nn.MaxPool2d):
            layers.append(l)
            l = []
        l.append(vgg.features[i])
    l += vgg_base_extra(bn=bn)
    layers.append(l)

    # layers of feature scaling 2**5
    block = ConvBnRelu if bn else ConvRelu
    layer5 = [block(1024, 256, 1, 1, 0),
              block(256, 512, 3, 2, 1)]
    layers.append(layer5)

    layers = [Sequential(*l) for l in layers]
    n_pretrained = 4 if imagenet_pretrained else 0
    return layers, bn, n_pretrained


def resnet(name, pretrained, **kwargs):
    abn = name.endswith('_abn')
    if abn:
        name = name[:-4]

    pool_in_2nd = name.startswith('R')
    name = name.lower()

    imagenet_pretrained = pretrained == 'imagenet'

    if 'replace_stride_with_dilation' in kwargs and name in ('resnet18', 'resnet34'):
        replace_stride_with_dilation = kwargs['replace_stride_with_dilation']
        warnings.warn(f"{name} ignore replace_stride_with_dilation={replace_stride_with_dilation}")
        del kwargs['replace_stride_with_dilation']

    if name == 'resnet18':
        resnet = torchvision.models.resnet18(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'resnet34':
        resnet = torchvision.models.resnet34(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'resnet50':
        resnet = torchvision.models.resnet50(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'resnet101':
        resnet = torchvision.models.resnet101(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'resnet152':
        resnet = torchvision.models.resnet152(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'resnet50_ibn_a':
        from .resnet_ibn_a import resnet50_ibn_a
        resnet = resnet50_ibn_a(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'resnet101_ibn_a':
        from .resnet_ibn_a import resnet101_ibn_a
        resnet = resnet101_ibn_a(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'resnext50_32x4d':
        resnet = torchvision.models.resnext50_32x4d(pretrained=imagenet_pretrained, **kwargs)
    elif name == 'resnext101_32x8d':
        if pretrained == 'instagram':
            resnet = resnext_wsl(name, pretrained=True, **kwargs)
        else:
            resnet = torchvision.models.resnext101_32x8d(pretrained=imagenet_pretrained, **kwargs)
    elif name in ('resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d'):
        resnet = resnext_wsl(name, pretrained=pretrained=='instagram', **kwargs)
    else:
        raise NotImplementedError(name)

    if not imagenet_pretrained and pretrained and pretrained.endswith('.pth'):
        state_dict = torch.load(pretrained)
        resnet.load_state_dict(state_dict, strict=False)

    if pool_in_2nd:
        layer0 = Sequential(resnet.conv1, resnet.bn1, resnet.relu)
    else:
        layer0 = Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

    layer0[-1].out_channels = resnet.bn1.num_features

    def get_out_channels_from_resnet_block(layer):
        block = layer[-1]
        if isinstance(block, torchvision.models.resnet.BasicBlock):
            return block.conv2.out_channels
        elif isinstance(block, (torchvision.models.resnet.Bottleneck, BottleneckIBNa)):
            return block.conv3.out_channels
        raise RuntimeError("unknown resnet block: {}".format(block.__class__))

    if abn:
        layer0 = replace_bn_in_sequential(layer0)
        block = resnet.layer1[0].__class__
        layer1 = replace_bn_in_sequential(resnet.layer1, block=block)
        layer2 = replace_bn_in_sequential(resnet.layer2, block=block)
        layer3 = replace_bn_in_sequential(resnet.layer3, block=block)
        layer4 = replace_bn_in_sequential(resnet.layer4, block=block)
    else:
        layer1 = resnet.layer1
        layer2 = resnet.layer2
        layer3 = resnet.layer3
        layer4 = resnet.layer4

    layer1.out_channels = layer1[-1].out_channels = get_out_channels_from_resnet_block(resnet.layer1)
    layer2.out_channels = layer2[-1].out_channels = get_out_channels_from_resnet_block(resnet.layer2)
    layer3.out_channels = layer3[-1].out_channels = get_out_channels_from_resnet_block(resnet.layer3)
    layer4.out_channels = layer4[-1].out_channels = get_out_channels_from_resnet_block(resnet.layer4)

    if pool_in_2nd:
        layer1 = nn.Sequential(resnet.maxpool, layer1)
        layer1.out_channels = layer1[-1].out_channels

    n_pretrained = 5 if imagenet_pretrained else 0

    return [layer0, layer1, layer2, layer3, layer4], True, n_pretrained


def resnext(name, pretrained):
    pool_in_2nd = name.startswith('R')
    name = name.lower()

    if name in ['resnext101_32x4d', 'resnext101_64x4d']:
        import pretrainedmodels
        imagenet_pretrained = 'imagenet' if pretrained == 'imagenet' else None
        resnext = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=imagenet_pretrained)
    else:
        return NotImplemented

    resnext_features = resnext.features
    layer0 = [resnext_features[i] for i in range(4)]
    if pool_in_2nd:
        layer0 = layer0[:-1]
    layer0 = nn.Sequential(*layer0)
    layer0.out_channels = layer0[-1].out_channels = 64

    layer1 = resnext_features[4]
    if pool_in_2nd:
        layer1 = nn.Sequential(resnext_features[3], layer1)
    layer1.out_channels = layer1[-1].out_channels = 256

    layer2 = resnext_features[5]
    layer2.out_channels = layer2[-1].out_channels = 512

    layer3 = resnext_features[6]
    layer3.out_channels = layer3[-1].out_channels = 1024

    layer4 = resnext_features[7]
    layer4.out_channels = layer4[-1].out_channels = 2048
    n_pretrained = 5 if imagenet_pretrained else 0

    return [layer0, layer1, layer2, layer3, layer4], True, n_pretrained


def resnext_wsl(arch, pretrained, progress=True, **kwargs):
    """
    models trained in weakly-supervised fashion on 940 million public images with 1.5K hashtags matching with 1000 ImageNet1K synsets, followed by fine-tuning on ImageNet1K dataset.
    https://github.com/facebookresearch/WSL-Images/
    """
    from torch.hub import load_state_dict_from_url
    from torchvision.models.resnet import ResNet, Bottleneck

    model_args = {'resnext101_32x8d': dict(block=Bottleneck, layers=[3, 4, 23, 3], groups=32, width_per_group=8),
                  'resnext101_32x16d': dict(block=Bottleneck, layers=[3, 4, 23, 3], groups=32, width_per_group=16),
                  'resnext101_32x32d': dict(block=Bottleneck, layers=[3, 4, 23, 3], groups=32, width_per_group=32),
                  'resnext101_32x48d': dict(block=Bottleneck, layers=[3, 4, 23, 3], groups=32, width_per_group=48)}

    args = model_args[arch]
    args.update(kwargs)
    model = ResNet(**args)

    if pretrained:
        model_urls = {
            'resnext101_32x8d': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
            'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
            'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
            'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
        }
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)

    return model


def replace_bn(bn, act=None):
    slop = 0.01
    if isinstance(act, nn.ReLU):
        activation = 'leaky_relu'  # approximate relu
    elif isinstance(act, nn.LeakyReLU):
        activation = 'leaky_relu'
        slope = act.negative_slope
    elif isinstance(act, nn.ELU):
        activation = 'elu'
    else:
        activation = 'none'
    abn = ActivatedBatchNorm(num_features=bn.num_features,
                             eps=bn.eps,
                             momentum=bn.momentum,
                             affine=bn.affine,
                             track_running_stats=bn.track_running_stats,
                             activation=activation,
                             slope=slop)
    abn.load_state_dict(bn.state_dict())
    return abn


def replace_bn_in_sequential(layer0, block=None):
    layer0_modules = []
    last_bn = None
    for n, m in layer0.named_children():
        if isinstance(m, nn.BatchNorm2d):
            last_bn = (n, m)
        else:
            activation = 'none'
            if last_bn:
                abn = replace_bn(last_bn[1], m)
                activation = abn.activation
                layer0_modules.append((last_bn[0], abn))
                last_bn = None
            if activation == 'none':
                if block and isinstance(m, block):
                    m = replace_bn_in_block(m)
                elif isinstance(m, nn.Sequential):
                    m = replace_bn_in_sequential(m, block)
                layer0_modules.append((n, m))
    if last_bn:
        abn = replace_bn(last_bn[1])
        layer0_modules.append((last_bn[0], abn))
    return nn.Sequential(OrderedDict(layer0_modules))


class DummyModule(nn.Module):
    def forward(self, x):
        return x


def replace_bn_in_block(block):
    block.bn1 = replace_bn(block.bn1, block.relu)
    block.bn2 = replace_bn(block.bn2, block.relu)
    block.bn3 = replace_bn(block.bn3)
    block.relu = DummyModule()
    if block.downsample:
        block.downsample = replace_bn_in_sequential(block.downsample)
    return nn.Sequential(block,
                         nn.ReLU(inplace=True))


def se_net(name, pretrained):
    pool_in_2nd = name.startswith('S')
    name = name.lower()

    abn = name.endswith('_abn')
    if abn:
        name = name[:-4]

    import pretrainedmodels
    if name in ['se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'senet154']:
        imagenet_pretrained = 'imagenet' if pretrained == 'imagenet' else None
        senet = pretrainedmodels.__dict__[name](num_classes=1000, pretrained=imagenet_pretrained)
    else:
        return NotImplemented

    layer0 = [senet.layer0[i] for i in range(len(senet.layer0))]
    if pool_in_2nd:
        layer0 = layer0[:-1]
    layer0 = nn.Sequential(*layer0)
    layer0 = replace_bn_in_sequential(layer0) if abn else layer0

    block = senet.layer1[0].__class__
    layer1 = replace_bn_in_sequential(senet.layer1, block=block) if abn else senet.layer1
    if pool_in_2nd:
        layer1 = nn.Sequential(senet.layer0[-1], layer1)
    layer1.out_channels = layer1[-1].out_channels = senet.layer1[-1].conv3.out_channels
    layer0.out_channels = layer0[-1].out_channels = senet.layer1[0].conv1.in_channels

    layer2 = replace_bn_in_sequential(senet.layer2, block=block) if abn else senet.layer2
    layer2.out_channels = layer2[-1].out_channels = senet.layer2[-1].conv3.out_channels

    layer3 = replace_bn_in_sequential(senet.layer3, block=block) if abn else senet.layer3
    layer3.out_channels = layer3[-1].out_channels = senet.layer3[-1].conv3.out_channels

    layer4 = replace_bn_in_sequential(senet.layer4, block=block) if abn else senet.layer4
    layer4.out_channels = layer4[-1].out_channels = senet.layer4[-1].conv3.out_channels

    n_pretrained = 5 if imagenet_pretrained else 0
    return [layer0, layer1, layer2, layer3, layer4], True, n_pretrained


def densenet(name, pretrained, **kwargs):
    imagenet_pretrained = 'imagenet' if pretrained == 'imagenet' else None
    if name == 'densenet121':
        net = torchvision.models.densenet121(pretrained=pretrained, **kwargs)
    else:
        raise NotImplementedError(name)

    n_pretrained = len(net.features) if imagenet_pretrained else 0
    return net.features, True, n_pretrained


def efficientnet(name, pretrained, in_channels=3, **kwargs):
    from efficientnet_pytorch import EfficientNet
    imagenet_pretrained = 'imagenet' if pretrained == 'imagenet' else None
    override_params = dict(image_size=None)
    if imagenet_pretrained:
        model = EfficientNet.from_pretrained(name.lower(), override_params=override_params)
    else:
        model = EfficientNet.from_name(name.lower(), override_params=override_params)

    if 'memory_efficient' in kwargs:
        if hasattr(model, 'set_swish'):
            model.set_swish(memory_efficient=kwargs['memory_efficient'])

    return convert_efficientnet(model, name, pretrained, in_channels)


def convert_efficientnet(model, name, pretrained, in_channels=3):
    from efficientnet_pytorch.model import MBConvBlock
    layers = []

    # set input channels
    conv_stem = model._conv_stem
    if in_channels != 3:
        conv_type = type(conv_stem)
        image_size = model._global_params.image_size
        conv_stem = conv_type(in_channels=in_channels, out_channels=conv_stem.out_channels,
                              kernel_size=conv_stem.kernel_size, stride=conv_stem.stride,
                              bias=conv_stem.bias, image_size=image_size)

    stem = Sequential(conv_stem, model._bn0, Swish(inplace=True))

    blocks = [stem]
    for idx, block in enumerate(model._blocks):
        if block._depthwise_conv.stride[0] == 2:
            l = nn.Sequential(*blocks)
            layers.append(l)
            blocks = []

        drop_connect_rate = model._global_params.drop_connect_rate
        if drop_connect_rate:
            drop_connect_rate *= float(idx) / len(model._blocks)
        block.drop_connect_rate = drop_connect_rate
        blocks.append(block)
    if blocks:
        layers.append(nn.Sequential(*blocks))

    layers += [Sequential(model._conv_head, model._bn1, Swish(inplace=True))]

    for l in layers:
        if isinstance(l[-1], MBConvBlock):
            l.out_channels = l[-1]._bn2.num_features
        else:
            l.out_channels = get_num_of_channels(l)

    if name[0] == 'E':
        # make sure output size are different
        layers_merged = nn.Sequential(layers[4], layers[5])
        layers_merged.out_channels = layers[5].out_channels
        layers = layers[:4] + [layers_merged]
    else:
        layers = layers[:5]

    n_pretrained = len(layers) if pretrained else 0
    return layers, True, n_pretrained


def darknet(pretrained):
    from .darknet import KitModel as DarkNet
    net = DarkNet()
    if pretrained:
        state_dict = model_zoo.load_url(get_model_zoo_url('darknet', 'coco'), map_location=lambda storage, loc: storage)
        net.load_state_dict(state_dict)
    n_pretrained = 3 if pretrained else 0
    return [net.model0, net.model1, net.model2], True, n_pretrained


def mobilenet_v2(pretrained):
    from .mobile_net_v2 import MobileNetV2
    net = MobileNetV2()
    if pretrained:
        state_dict = model_zoo.load_url(get_model_zoo_url('mobilenet_v2', pretrained), map_location=lambda storage, loc: storage)
        net.load_state_dict(state_dict)
        pretrained = True
    else:
        pretrained = False

    splits = [0, 2, 4, 7, 14]
    layers = [net.features[i:j] for i, j in zip(splits, splits[1:] + [len(net.features)-1])]
    for l in layers:
        l.out_channels = l[-1].out_channels

    n_pretrained = len(layers) if pretrained else 0
    return layers, True, n_pretrained


def torch_vision_mobilenet_v2(pretrained):
    from torchvision.models import mobilenet_v2
    imagenet_pretrained = pretrained == 'imagenet'
    net = mobilenet_v2(imagenet_pretrained)
    splits = [0, 2, 4, 7, 14]
    layers = [net.features[i:j] for i, j in zip(splits, splits[1:] + [len(net.features)-1])]
    for l in layers:
        l.out_channels = l[-1].conv[-2].out_channels

    n_pretrained = len(layers) if pretrained else 0
    return layers, True, n_pretrained


def mobilenet_v3(pretrained):
    imagenet_pretrained = pretrained == 'imagenet'
    net = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'mobilenetv3_rw', pretrained=imagenet_pretrained)

    stem = Sequential(net.conv_stem, net.bn1, Swish())
    layers = [stem]

    splits = [0, 2, 3, 5, 7]
    blocks = [net.blocks[i:j] for i, j in zip(splits[:-1], splits[1:])]
    for block, out_channels in zip(blocks, [24, 40, 112, 960]):
        block.out_channels = out_channels
        layers.append(block)

    n_pretrained = len(layers) if pretrained else 0
    return layers, True, n_pretrained


def shufflenet_v2(name, pretrained):
    from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
    imagenet_pretrained = pretrained == 'imagenet'

    if name == 'shufflenet_v2_x0_5':
        net = shufflenet_v2_x0_5(imagenet_pretrained)
    elif name == 'shufflenet_v2_x1_0':
        net = shufflenet_v2_x1_0(imagenet_pretrained)
    elif name == 'shufflenet_v2_x1_5':
        net = shufflenet_v2_x1_5(imagenet_pretrained)
    elif name == 'shufflenet_v2_x2_0':
        net = shufflenet_v2_x2_0(imagenet_pretrained)
    else:
        raise NotImplementedError(name)

    stage1 = nn.Sequential(net.conv1, net.maxpool)
    layers = [stage1, net.stage2, net.stage3, net.stage4, net.conv5]
    for stage, out in zip(layers, net._stage_out_channels):
        stage.out_channels = out

    n_pretrained = len(layers) if imagenet_pretrained else 0
    return layers, True, n_pretrained


def squeezenet1(name, pretrained):
    from torchvision.models import squeezenet1_0, squeezenet1_1
    imagenet_pretrained = pretrained == 'imagenet'

    if name == 'squeezenet1_0':
        net = squeezenet1_0(imagenet_pretrained)
        splits = [0, 3, 7, 12, 13]
        out_channels = [96, 256, 512, 512]
    elif name == 'squeezenet1_1':
        net = squeezenet1_1(imagenet_pretrained)
        splits = [0, 3, 6, 9, 13]
        out_channels = [64, 128, 256, 512]
    else:
        raise NotImplementedError(name)

    layers = [net.features[i:j] for i, j in zip(splits[:-1], splits[1:])]
    for block, o in zip(layers, out_channels):
        block.out_channels = o

    n_pretrained = len(layers) if pretrained else 0
    return layers, True, n_pretrained


class MockModule(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.backbone = nn.ModuleList(layers)


def load_pretrained_weights(layers, name, dataset_name):
    state_dict = model_zoo.load_url(get_model_zoo_url(name, dataset_name))
    mock_module = MockModule(layers)
    mock_module.load_state_dict(state_dict, strict=False)


def create_basenet(name, pretrained, drop_last=0, activation=None, frozen_batchnorm=False, in_channels=3, **kwargs):
    """
    Parameters
    ----------
    name: model name
    pretrained: dataset name

    Returns
    -------
    list of modules, is_batchnorm, num_of_pretrained_module
    """
    if name.startswith('vgg'):
        layers, bn, n_pretrained = vgg(name, pretrained)
    elif name.lower() in ('resnext101_32x4d', 'resnext101_64x4d'):
        layers, bn, n_pretrained = resnext(name, pretrained)
    elif name.lower().startswith('resnet') or name.lower().startswith('resnext'):
        layers, bn, n_pretrained = resnet(name, pretrained, **kwargs)
    elif name.lower().startswith('se'):
        layers, bn, n_pretrained = se_net(name, pretrained)
    elif name.lower().startswith('densenet'):
        layers, bn, n_pretrained = densenet(name, pretrained, **kwargs)
    elif name.lower().startswith('efficientnet'):
        layers, bn, n_pretrained = efficientnet(name, pretrained, in_channels=in_channels, memory_efficient=True)
    elif name == 'darknet':
        layers, bn, n_pretrained = darknet(pretrained)
    elif name == 'mobilenet_v2':
        layers, bn, n_pretrained = mobilenet_v2(pretrained)
    elif name == 'Mobilenet_v2':
        layers, bn, n_pretrained = torch_vision_mobilenet_v2(pretrained)
    elif name == 'mobilenet_v3':
        layers, bn, n_pretrained = mobilenet_v3(pretrained)
    elif name.startswith('shufflenet_v2'):
        layers, bn, n_pretrained = shufflenet_v2(name, pretrained)
    elif name.startswith('squeezenet1'):
        layers, bn, n_pretrained = squeezenet1(name, pretrained)
    else:
        raise NotImplemented(name)

    if pretrained in ('voc', 'coco', 'oid'):
        load_pretrained_weights(layers, name, pretrained)
        n_pretrained = len(layers)

    if drop_last > 0:
        layers = layers[:-drop_last]
        n_pretrained = max(0, n_pretrained - drop_last)

    if activation:
        layers = [convert_activation(activation, l) for l in layers]

    if frozen_batchnorm:
        from .batch_norm import FrozenBatchNorm2d
        layers = [FrozenBatchNorm2d.convert_frozen_batchnorm(l) for l in layers]

    return layers, bn, n_pretrained

