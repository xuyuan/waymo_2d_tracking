import torch
from torch import nn as nn
from torch.nn.functional import softmax, binary_cross_entropy, binary_cross_entropy_with_logits, one_hot

from .cross_entropy import cross_entropy


def reduce_loss(loss, reduction):
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return torch.sum(loss)
    elif reduction == 'mean':
        return loss.mean()
    else:
        raise NotImplementedError(reduction)


def focal_loss(input, target, focusing=2, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None):
    """
    focusing (float): parameter of Focal Loss (0 equals standard cross entropy)
    other parameters: see http://pytorch.org/docs/master/nn.html#torch.nn.functional.cross_entropy
    """
    l = cross_entropy(input, target, weight=weight, ignore_index=ignore_index, reduction='none', smooth_eps=smooth_eps)
    if input.dim() == 4:
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, input.size(1))

    p = softmax(input, dim=1)
    masked_indices = target.ne(ignore_index)
    target = target[masked_indices]
    p = p[masked_indices]

    p = p.gather(-1, target.view(-1, 1)).squeeze()
    if l.dim() != p.dim():
        p = p.view(*l.size())

    fl = l * ((1 - p) ** focusing)
    return reduce_loss(fl, reduction)


def focal_loss_sigmoid(input, target, focusing=2, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None):
    """
    Args:
        input: N, C
        target: N
    """
    masked_indices = target.ne(ignore_index)
    target = target[masked_indices]
    input = input[masked_indices]

    num_classes = input.size(-1)
    target = one_hot(target, num_classes=num_classes+1)[..., 1:]

    if weight is not None:
        weight = weight.unsqueeze(0)
        weight = target * weight + (1 - target) * (1 - weight)

    smooth_eps = smooth_eps or 0
    if smooth_eps > 0:
        target_float = target * (1 - smooth_eps) + (1 - target) * smooth_eps
    else:
        target_float = target.to(input)

    if focusing > 0:
        p = torch.sigmoid(input)
        # neg_p = p * (1 - target) + (1 - p) * target
        neg_p = p - p * target * 2 + target

        l = binary_cross_entropy_with_logits(input, target_float, weight=weight, reduction='none')
        fl = l * (neg_p ** focusing)
        return reduce_loss(fl, reduction)
    else:
        return binary_cross_entropy_with_logits(input, target_float, weight=weight, reduction=reduction)


class FocalLoss(nn.Module):
    def __init__(self, focusing, weight=None, reduction='mean', smooth_eps=None):
        super().__init__()
        self.focusing = focusing
        if weight is not None:
            self.weight = nn.Parameter(weight, requires_grad=False)
        else:
            self.weight = None
        self.reduction = reduction
        self.smooth_eps = smooth_eps

    def forward(self, input, target):
        if self.focusing > 0:
            return focal_loss(input, target,
                              self.focusing,
                              weight=self.weight,
                              reduction=self.reduction,
                              smooth_eps=self.smooth_eps)
        else:
            return cross_entropy(input, target,
                                 weight=self.weight,
                                 reduction=self.reduction,
                                 smooth_eps=self.smooth_eps)


class FocalLossSigmoid(FocalLoss):
    def forward(self, input, target):
        return focal_loss_sigmoid(input, target,
                                  self.focusing,
                                  weight=self.weight,
                                  reduction=self.reduction,
                                  smooth_eps=self.smooth_eps)