import torch.nn.functional as F


def cross_entropy(inputs, target, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567
    based on https://github.com/eladhoffer/utils.pytorch/blob/master/cross_entropy.py
    """
    smooth_eps = smooth_eps or 0

    # ordinary log-liklihood - use cross_entropy from nn
    if smooth_eps == 0:
        loss = F.cross_entropy(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
        if reduction == 'none':
           masked_indices = target.ne(ignore_index)
           loss = loss[masked_indices]
        return loss

    lsm = F.log_softmax(inputs, dim=-1)

    num_classes = inputs.size(-1)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    masked_indices = target.ne(ignore_index)
    lsm = lsm[masked_indices]
    target = target[masked_indices]

    eps_sum = smooth_eps / num_classes
    eps_nll = 1. - eps_sum - smooth_eps
    likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))

    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        loss = loss.mean()
    return loss
