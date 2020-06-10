"""
Stochastic Weight Averaging (SWA)

Averaging Weights Leads to Wider Optima and Better Generalization

https://github.com/timgaripov/swa
"""
from pathlib import Path
import warnings
import argparse

import torch
from torch.utils.data import DataLoader
from .utils import choose_device, get_num_workers
from tqdm import tqdm


def moving_average(net1, net2, alpha=1.):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, device):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        :param dataset: train dataset for buffers average estimation.
        :param model: model being update
        :param jobs: jobs for dataloader
        :return: None
    """
    if not check_bn(model):
        print('no bn in model?!')
        return model

    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0

    model = model.to(device)
    pbar = tqdm(loader, unit="samples", unit_scale=loader.batch_size)
    for sample in pbar:
        if isinstance(sample, dict):
            inputs = sample.get('input', None)
        elif isinstance(sample, (list, tuple)):
            inputs, _ = sample
        elif torch.is_tensor(sample):
            inputs = sample
        else:
            inputs = None

        if inputs is None:
            warnings.warn("empty inputs")
            continue

        inputs = inputs.to(device)
        b = inputs.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(inputs)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
    return model


def swa(load_model, input_dir, output, device, dataloader=None):
    output = Path(output)
    if output.exists():
        raise RuntimeError(f'output file exists: {output}')

    directory = Path(input_dir)
    model_files = [f for f in directory.iterdir() if str(f).endswith(".model.pth")]
    assert(len(model_files) > 1)

    device = choose_device(device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    net = load_model(model_files[0])
    for i, f in enumerate(model_files[1:]):
        net2 = load_model(f)
        moving_average(net, net2, 1. / (i + 2))

    if dataloader:
        with torch.no_grad():
            net = bn_update(dataloader, net, device)
    else:
        if check_bn(net):
            # raise RuntimeError("Please update BN!")
            warnings.warn("Please update BN!")

    output.parent.mkdir(parents=True, exist_ok=True)
    net.save(str(output))

    return net


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, description=__doc__):
        super().__init__(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.add_argument("-i", "--input", type=str, help='input directory which contains models')
        self.add_argument("-o", "--output", type=str, default='swa_model.pth', help='output model file')
        self.add_argument("--batch-size", type=int, default=8, help='batch size')
        self.add_argument('--device', default='auto', choices=['cuda', 'cpu'], help='running with cpu or cuda')
        self.add_argument('-j', '--jobs', default=-2, type=int, help='How many subprocesses to use for data loading. ' +
                                                                       'Negative or 0 means number of cpu cores left')