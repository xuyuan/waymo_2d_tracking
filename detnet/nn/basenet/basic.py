from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .inplace_abn import ActivatedBatchNorm
from .tf_like import MaxPool2dSame, Conv2dSame


def Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
    if padding == 'same':
        if padding_mode != 'zeros':
            raise NotImplementedError(f"padding_mode {padding_mode} for same padding")
        return Conv2dSame(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                          stride=stride, dilation=dilation, groups=groups, bias=bias)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)


def MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
    if padding == 'same':
        return MaxPool2dSame(kernel_size=kernel_size, stride=stride, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)
    return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode)


def conv(*args, **kwargs):
    return lambda last_layer: Conv2d(get_num_of_channels(last_layer), *args, **kwargs)


def get_num_of_channels(layers, channle_name='out_channels'):
    """access out_channels from last layer of nn.Sequential/list"""
    if hasattr(layers, channle_name):
        return getattr(layers, channle_name)
    elif isinstance(layers, int):
        return layers
    elif isinstance(layers, nn.BatchNorm2d):
        return layers.num_features
    else:
        for i in range(len(layers) - 1, -1, -1):
            layer = layers[i]
            if hasattr(layer, channle_name):
                return getattr(layer, channle_name)
            elif isinstance(layer, nn.Sequential):
                return get_num_of_channels(layer, channle_name)
    raise RuntimeError("cant get_num_of_channels {} from {}".format(channle_name, layers))


def Sequential(*args):
    f = nn.Sequential(*args)
    f.in_channels = get_num_of_channels(f, 'in_channels')
    f.out_channels = get_num_of_channels(f)
    return f


def sequential(*args):
    def create_sequential(last_layer):
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            layers = OrderedDict()
            for key, a in args[0].items():
                m = a(last_layer)
                layers[key] = m
                last_layer = m
            return Sequential(layers)
        else:
            layers = []
            for a in args:
                layers.append(a(last_layer))
                last_layer = layers[-1]
            return Sequential(*layers)
    return create_sequential


def ConvBn(*args, **kwargs):
    """drop in block for Conv2d with BatchNorm and ReLU"""
    c = Conv2d(*args, **kwargs)
    return Sequential(c,
                      nn.BatchNorm2d(c.out_channels))


def conv_bn(*args, **kwargs):
    return lambda last_layer: ConvBn(get_num_of_channels(last_layer), *args, **kwargs)


def ConvBnRelu(*args, **kwargs):
    """drop in block for Conv2d with BatchNorm and ReLU"""
    c = Conv2d(*args, **kwargs)
    return Sequential(c,
                      nn.BatchNorm2d(c.out_channels),
                      nn.ReLU(inplace=True))


def conv_bn_relu(*args, **kwargs):
    return lambda last_layer: ConvBnRelu(get_num_of_channels(last_layer), *args, **kwargs)


def ConvBnRelu6(*args, **kwargs):
    """drop in block for Conv2d with BatchNorm and ReLU6"""
    c = Conv2d(*args, **kwargs)
    return Sequential(c,
                      nn.BatchNorm2d(c.out_channels),
                      nn.ReLU(inplace=True))


def conv_bn_relu6(*args, **kwargs):
    return lambda last_layer: ConvBnRelu6(get_num_of_channels(last_layer), *args, **kwargs)


def ConvRelu(*args, **kwargs):
    return Sequential(Conv2d(*args, **kwargs),
                      nn.ReLU(inplace=True))


def conv_relu(*args, **kwargs):
    return lambda last_layer: ConvRelu(get_num_of_channels(last_layer), *args, **kwargs)


def ConvRelu6(*args, **kwargs):
    return Sequential(Conv2d(*args, **kwargs),
                      nn.ReLU6(inplace=True))


def conv_relu6(*args, **kwargs):
    return lambda last_layer: ConvRelu6(get_num_of_channels(last_layer), *args, **kwargs)


def ReluConv(*args, **kwargs):
    return Sequential(nn.ReLU(inplace=True),
                      Conv2d(*args, **kwargs))


def relu_conv(*args, **kwargs):
    return lambda last_layer: ReluConv(get_num_of_channels(last_layer), *args, **kwargs)


def BnReluConv(*args, **kwargs):
    """drop in block for Conv2d with BatchNorm and ReLU"""
    c = Conv2d(*args, **kwargs)
    return Sequential(nn.BatchNorm2d(c.in_channels),
                      nn.ReLU(inplace=True),
                      c)


def bn_relu_conv(*args, **kwargs):
    return lambda last_layer: BnReluConv(get_num_of_channels(last_layer), *args, **kwargs)


def maxpool(*args, **kwargs):
    def max_pool_module(last_layer):
        m = MaxPool2d(*args, **kwargs)
        m.in_channels = m.out_channels = last_layer.out_channels
        return m
    return max_pool_module


def SeparableConv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, padding_mode='zeros', depth_multiplier=1):
    m = OrderedDict()
    n_features = in_channels * depth_multiplier
    m['depthwise'] = Conv2d(in_channels=in_channels, out_channels=n_features, kernel_size=kernel_size, stride=stride,
                            padding=padding, groups=in_channels, bias=False, padding_mode=padding_mode)
    m['pointwise'] = Conv2d(in_channels=n_features, out_channels=out_channels, kernel_size=1, bias=bias)
    return Sequential(m)


def separable_conv2d(*args, **kwargs):
    return lambda last_layer: SeparableConv2d(get_num_of_channels(last_layer), *args, **kwargs)


def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    """
    Checkerboard artifact free sub-pixel convolution: https://arxiv.org/abs/1707.02937

    code from https://github.com/fastai/fastai/blob/master/fastai/layers.py#L222

    :param x: torch.Tensor, e.g. `Conv2d.weight`
    :param scale: factor to increase spatial resolution
    :param init: initializer to be used for sub_kernel initialization
    """
    ni, nf, h, w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2, nf, h ,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
    x.data.copy_(k)


def create_norm(name, layers):
    if not name:
        return

    n_feats = layers[-1].out_channels
    if name == 'instance_norm':
        layers.append(nn.InstanceNorm2d(n_feats, affine=True))
    elif name == 'batch_norm':
        layers.append(nn.BatchNorm2d(n_feats))
    elif name == 'weight':
        layers[-1] = nn.utils.weight_norm(layers[-1])
    else:
        raise NotImplementedError(name)


class SubPixelConv(nn.Sequential):
    def __init__(self, scale, in_channels, kernel_size=3, bias=True, norm=None, activation=None, icnr_init=True, blur=False, out_channels=None):
        if out_channels is None:
            out_channels = in_channels
        conv = Conv2d(in_channels=in_channels, out_channels=out_channels * 2 ** scale, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=bias)
        if icnr_init:
            icnr(conv.weight, scale)
        layers = [conv]
        create_norm(norm, layers)
        layers += [nn.PixelShuffle(scale)]
        if activation:
            layers += [activation]

        if blur:
            # Blurring over (h*w) kernel
            # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
            # - https://arxiv.org/abs/1806.02658
            layers += [nn.ReplicationPad2d((1, 0, 1, 0)),
                       nn.AvgPool2d(2, stride=1)]

        super().__init__(*layers)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    def __init__(self, inplace=False, memory_efficient=True):
        super().__init__()
        self.inplace = inplace
        self.memory_efficient = memory_efficient

    def forward(self, x):
        if self.memory_efficient:
            SwishImplementation.apply(x)
        elif self.inplace:
            return x.mul_(torch.sigmoid(x))
        return x * torch.sigmoid(x)

    def extra_repr(self):
        inplace_str = []
        if self.inplace:
            inplace_str.append('inplace=True')
        if self.memory_efficient:
            inplace_str.append('memory_efficient=True')

        return ', '.join(inplace_str)


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.mul_(F.relu6(x + 3.) / 6.)
        else:
            return x * F.relu6(x + 3.) / 6.

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class Mish(nn.Module):
    """
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    from https://github.com/digantamisra98/Mish/
    """
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


def convert_activation(activation, module):
    if isinstance(activation, str):
        act_cls = {'mish': Mish,
                   'swish': Swish,
                   'relu': nn.ReLU,
                   'hardswish': HardSwish}
        activation = act_cls[activation.lower()]

    module_output = module
    if module.__class__.__name__ in ('ReLU', 'Swish', 'Mish'):
        module_output = activation(getattr(module, 'inplace', False))
    elif isinstance(module, ActivatedBatchNorm) and module_output.activation != 'none':
        module_output.activation = activation.__name__.lower()
    for name, child in module.named_children():
        module_output.add_module(name, convert_activation(activation, child))
    del module
    return module_output


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=(1, 1)):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class InputNormalization(nn.Module):
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=True):
        super().__init__()
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
            return x
        else:
            return x.sub(self.mean[None, :, None, None]).div(self.std[None, :, None, None])

