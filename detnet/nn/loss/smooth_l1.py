"""
very similar to the smooth_l1_loss from pytorch, but with the extra beta parameter

from maskrcnn_benchmark/layers/smooth_l1_loss.py
"""

import torch


class SmoothL1Loss(torch.nn.Module):
    def __init__(self, beta=1. / 9, reduction='mean'):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, input, target):
        return smooth_l1_loss(input, target, self.beta, self.reduction)


def smooth_l1_loss(input, target, beta=1. / 9, reduction='mean'):
    n = torch.abs(input - target)
    cond = n < beta
    ret = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


class AdjustSmoothL1Loss(SmoothL1Loss):
    def __init__(self, num_features, momentum=0.1, beta=1. / 9, reduction='mean'):
        super().__init__(beta=beta, reduction=reduction)
        self.num_features = num_features
        self.momentum = momentum
        self.beta = beta
        self.register_buffer(
            'running_mean', torch.empty(num_features).fill_(beta)
        )
        self.register_buffer('running_var', torch.zeros(num_features))

    def forward(self, inputs, target):
        n = torch.abs(inputs - target)
        with torch.no_grad():
            if torch.isnan(n.var(dim=0)).sum().item() == 0:
                self.running_mean = self.running_mean.to(n.device)
                self.running_mean *= (1 - self.momentum)
                self.running_mean += (self.momentum * n.mean(dim=0))
                self.running_var = self.running_var.to(n.device)
                self.running_var *= (1 - self.momentum)
                self.running_var += (self.momentum * n.var(dim=0))

        beta = (self.running_mean - self.running_var)
        beta = beta.clamp(max=self.beta, min=1e-3)

        beta = beta.view(-1, self.num_features).to(n.device)
        cond = n < beta.expand_as(n)
        ret = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret
