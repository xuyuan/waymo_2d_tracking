import math
from torch import nn
from torch.nn.functional import relu
from ..basenet.basic import Swish


class MultiLevelHead(nn.ModuleList):
    def __init__(self, modules):
        super().__init__(modules)
        self.apply(weights_normal_init)

    def forward(self, features):
        return [m(x) for m, x in zip(self, features)]

    def bias_init_with_prior_prob(self, activation, p=1e-5):
        for m in self:
            bias_init_with_prior_prob(m, p=p, activation=activation)


class SharedHead(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.module = modules[0]

        # make sure we can unify, e.g. all modules has same i/o num of channels
        branch0 = repr(modules[0])
        for i in range(1, len(modules)):
            branch_i = repr(modules[i])
            if branch0 != branch_i:
                raise RuntimeError(f'{branch0} != {branch_i}')

        self.apply(weights_normal_init)

    def forward(self, features):
        return [self.module(x) for x in features]

    def bias_init_with_prior_prob(self, activation, p=1e-5):
        bias_init_with_prior_prob(self.module[-1], p=p, activation=activation)


# The convolution layers in the prediction net are shared among all levels, but
# each level has its batch normalization to capture the statistical
# difference among different levels.
class EfficientHead(SharedHead):
    def __init__(self, modules, survival_prob=None):
        """
        Args:
            modules: [[features], pred_head] * levels
            survival_prob:
        """
        super().__init__(modules)
        levels = len(modules)

        self.bns = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(c.out_channels, momentum=0.01, eps=1e-3) for c in self.module[0]])
                                  for l in range(levels)])
        self.act = Swish()
        self.survival_prob = survival_prob

    def _forward(self, x, bns):
        for i, (m, bn) in enumerate(zip(self.module[0], bns)):
            res = x
            x = m(x)
            x = bn(x)
            x = self.act(x)
            if i > 0 and self.survival_prob:
                x = self.drop_connect(x)
                x += res

        x = self.module[1](x)
        return x

    def drop_connect(self, x):
        """Drop the entire conv with given survival probability."""
        # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
        if not self.training:
            return x

        # Compute tensor.
        batch_size = x.shape[0]
        random_tensor = x.new_empty([batch_size, 1, 1, 1])
        random_tensor.uniform_()  # [0, 1]
        random_tensor += self.survival_prob  # [p, 1+p]
        random_tensor.floor_()  # 0 or 1
        # Unlike conventional way that multiply survival_prob at test time, here we
        # divide survival_prob at training time, such that no addition compute is
        # needed at test time.
        output = x / self.survival_prob * random_tensor
        return output

    def forward(self, features):
        return [self._forward(x, bns) for x, bns in zip(features, self.bns)]


def weights_normal_init(module, mean=0, std=0.01, bias=0):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)


def bias_init_with_prior_prob(conf_module, p=1e-5, activation='softmax'):
    # prior confidence for objects, see https://arxiv.org/abs/1708.02002
    # adjusted parameter according to https://arxiv.org/pdf/1909.04868.pdf
    if isinstance(conf_module, nn.Sequential):
        return bias_init_with_prior_prob(conf_module[-1], p=p, activation=activation)

    if isinstance(conf_module, nn.Conv2d):
        conf_bias = -math.log((1-p)/p)
        nn.init.constant_(conf_module.bias, conf_bias)

        if activation == 'softmax':
            # bias for background
            num_classes = len(conf_module.bias) - 1
            conf_module.bias.data[0] = math.log((1 - p * num_classes) / (p * num_classes))
    else:
        raise RuntimeError("Unknown type {0}".format(type(conf_module)))

