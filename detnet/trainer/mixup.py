"""
Mixed Sample Data Augmentation: MixUp, CutMix, FMix

mixup: Beyond Empirical Risk Minimization
Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
https://arxiv.org/abs/1710.09412

Understanding and Enhancing Mixed Sample Data Augmentation
Ethan Harris, Antonia Marcu, Matthew Painter, Mahesan Niranjan, Adam Prügel-Bennett, Jonathon Hare
https://arxiv.org/abs/2002.12047
https://github.com/ecs-vlc/FMix
"""

import functools
import numpy as np
import torch
from .transforms.vision import f_mask


def index_targets(targets, index):
    if isinstance(targets, torch.Tensor):
        targets = targets[index]
    elif isinstance(targets, list):
        targets = [targets[i] for i in index]
    elif isinstance(targets, dict):
        targets = {k: index_targets(v, index) for k, v in targets.items()}
    else:
        raise NotImplementedError(type(targets))
    return targets


def mixed_criterion(criterion, index, weight, outputs, targets):
    losses1 = criterion(outputs, targets)
    targets2 = index_targets(targets, index)
    losses2 = criterion(outputs, targets2)
    if torch.is_tensor(losses1):
        losses = weight * losses1 + (1 - weight) * losses2
    else:
        losses = {k: weight * losses1[k] + (1 - weight) * losses2[k] for k in losses1}
    return losses


def mixup(inputs, alpha, criterion):
    s = inputs.size(0)
    if s <= 1:
        return inputs, criterion

    weight = np.random.beta(alpha, alpha)
    index = np.random.permutation(s)
    x1, x2 = inputs, inputs[index]
    inputs = weight * x1 + (1 - weight) * x2

    return inputs, functools.partial(mixed_criterion, criterion, index, weight)


def cut_mix(inputs, alpha, criterion):
    s = inputs.size(0)
    if s <= 1:
        return inputs, criterion

    weight = np.random.beta(alpha, alpha)
    index = np.random.permutation(s)
    x1, x2 = inputs, inputs[index]
    cut_rat = (1. - weight) ** (1 / (x1.dim() - 2))

    rr = [rand_range(x1.size(i), cut_rat) for i in range(2, x1.dim())]
    if len(rr) == 1:
        x1[..., rr[0]] = x2[..., rr[0]]
    elif len(rr) == 2:
        x1[..., rr[0], rr[1]] = x2[..., rr[0], rr[1]]
    elif len(rr) == 3:
        x1[..., rr[0], rr[1], rr[2]] = x2[..., rr[0], rr[1], rr[2]]
    else:
        raise NotImplementedError(x1.dim())

    # adjust lambda to exactly match pixel ratio
    cut_rat = 1.0
    for r in rr:
        cut_rat *= (r.stop - r.start)
    for s in x1.shape[2:]:
        cut_rat /= s
    weight = 1 - cut_rat

    return x1, functools.partial(mixed_criterion, criterion, index, weight)


def rand_range(size, ratio):
    cut = int(ratio * ratio) // 2
    # uniform
    c = np.random.randint(size)

    l = np.clip(c - cut, 0, size)
    h = np.clip(c + cut, 0, size)
    return slice(l, h)


def f_mix(inputs, alpha, criterion, decay_power=3):
    # Sample mask and generate random permutation
    s = inputs.size(0)
    if s <= 1:
        return inputs, criterion

    size = inputs.shape[2:]
    weight = np.random.beta(alpha, alpha)
    index = np.random.permutation(s)
    x1, x2 = inputs, inputs[index]

    # Make mask
    mask = f_mask(weight, size, decay_power)
    mask = torch.from_numpy(mask).to(inputs)

    # Mix the images
    x = mask * x1 + (1 - mask) * x2
    return x, functools.partial(mixed_criterion, criterion, index, weight)



