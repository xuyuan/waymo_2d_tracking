import torch
from .lr_scheduler import FindLR, NoamLR, WarmUpLR


def create_optimizer(device, net, criterion, name, learning_rate, weight_decay, momentum=0, apex_opt_level=None,
                     optimizer_state=None, no_bn_wd=False, local_rank=None, sync_bn=False,
                     lr_scheduler_args=None, lr_scheduler_state=None, lookahead=False,
                     amp_state=None, cudnn_benchmark=True):
    net.float()

    if sync_bn and local_rank is not None:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    print('use', device)
    net = net.to(device)
    if isinstance(criterion, torch.nn.Module):
        criterion = criterion.to(device)

    param_groups = list(net.parameter_groups()) if hasattr(net, 'parameter_groups') else list(net.parameters())
    if len(param_groups) == 0:
        raise ValueError("optimizer got an empty parameter list")
    if not isinstance(param_groups[0], dict):
        param_groups = [{'params': param_groups}]

    parameters = []
    for param_group in param_groups:
        params = [p for p in param_group['params'] if p.requires_grad]
        if no_bn_wd:
            bn_params, remaining_params = bnwd_optim_params(net, params)
            if 'lr' in param_group:
                lr = param_group['lr']
                bn_params['lr'] = lr
                remaining_params['lr'] = lr
            parameters.append(bn_params)
            parameters.append(remaining_params)
        else:
            param_group['params'] = params
            parameters.append(param_group)

    # optimizer
    if name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif name == 'adamw':
        from torch.optim.adamw import AdamW
        optimizer = AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif name == 'RMSprop':
        from torch.optim.rmsprop import RMSprop
        optimizer = RMSprop(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif name == 'radam':
        from .radam import RAdam
        optimizer = RAdam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif name == 'fused_adam':
        from apex.optimizers import FusedAdam
        optimizer = FusedAdam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif name == 'adabound':
        from .adabound import AdaBound
        optimizer = AdaBound(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif name == 'adaboundw':
        from .adabound import AdaBoundW
        optimizer = AdaBoundW(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif name == 'novograd':
        from .novograd import Novograd
        optimizer = Novograd(parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise NotImplementedError(name)

    if lookahead:
        from .lookahead import Lookahead
        optimizer = Lookahead(optimizer)

    if apex_opt_level:
        from apex import amp
        net, optimizer = amp.initialize(net, optimizer, opt_level=apex_opt_level)

        def backward(loss):
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

        optimizer.backward = backward

        if isinstance(criterion, torch.nn.Module) and apex_opt_level == 'O1':
            criterion = criterion.half()
    else:
        optimizer.backward = lambda loss: loss.backward()

    lr_scheduler = create_lr_scheduler(optimizer, **lr_scheduler_args)

    if optimizer_state:
        # if use_fp16 and 'optimizer_state_dict' not in optimizer_state:
        #     # resume FP16_Optimizer.optimizer only
        #     optimizer.optimizer.load_state_dict(optimizer_state)
        # elif not use_fp16 and 'optimizer_state_dict' in optimizer_state:
        #     # resume optimizer from FP16_Optimizer.optimizer
        #     optimizer.load_state_dict(optimizer_state['optimizer_state_dict'])
        # else:
        optimizer.load_state_dict(optimizer_state)

    if lr_scheduler_state:
        lr_scheduler.load_state_dict(lr_scheduler_state)
        #print(f'resumed lr_scheduler_state{lr_scheduler.state_dict()}')
        #print(f"resumed learning rate {optimizer.param_groups[0]['lr']}")

    if apex_opt_level and amp_state:
        from apex import amp
        amp.load_state_dict(amp_state)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if local_rank is not None:
            # if apex_opt_level:
            #     import apex
            #     net = apex.parallel.DistributedDataParallel(net, delay_allreduce=True)
            # else:
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)
        else:
            net = torch.nn.DataParallel(net)

    return net, optimizer, lr_scheduler, criterion


# Filter out batch norm parameters and remove them from weight decay - gets us higher accuracy 93.2 -> 93.48
# https://arxiv.org/pdf/1807.11205.pdf
# code adopted from https://github.com/diux-dev/imagenet18/blob/master/training/experimental_utils.py
def bnwd_optim_params(model, model_params):
    bn_params, remaining_params = split_bn_params(model, model_params)
    return [{'params': bn_params, 'weight_decay': 0}, {'params': remaining_params}]


def split_bn_params(model, model_params):
    def get_bn_params(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm): return module.parameters()
        accum = set()
        for child in module.children(): [accum.add(p) for p in get_bn_params(child)]
        return accum

    mod_bn_params = get_bn_params(model)

    bn_params = [p for p in model_params if p in mod_bn_params]
    rem_params = [p for p in model_params if p not in mod_bn_params]
    return bn_params, rem_params


def create_lr_scheduler(optimizer, lr_scheduler, **kwargs):
    lr_scheduler_name = lr_scheduler

    if not isinstance(optimizer, torch.optim.Optimizer):
        # assume FP16_Optimizer
        optimizer = optimizer.optimizer

    eta_min = kwargs['min_learning_rate']
    max_epochs = kwargs['max_epochs']
    warmup_steps = kwargs['lr_scheduler_warmup']
    if 0 < warmup_steps < 1:
        warmup_steps = max(int(warmup_steps * max_epochs), 1)
    else:
        warmup_steps = int(warmup_steps)

    step_size = kwargs['lr_scheduler_step_size']
    step_size = [int(s * max_epochs) if s <= 1 else int(s) for s in step_size]  # scale according to max_epochs

    if lr_scheduler_name == 'plateau':
        patience = kwargs.get('lr_scheduler_patience', 10) // kwargs.get('validation_interval', 1)
        factor = kwargs.get('lr_scheduler_gamma', 0.1)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor, eps=0)
    elif lr_scheduler_name == 'step':
        gamma = kwargs.get('lr_scheduler_gamma', 0.1)
        if len(step_size) == 1:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size[0], gamma=gamma)
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=step_size, gamma=gamma)
    elif lr_scheduler_name == 'cos':
        assert len(step_size) == 1
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size[0], eta_min=eta_min)
    elif lr_scheduler_name == 'cosw':
        assert len(step_size) == 1
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=step_size[0], eta_min=eta_min)
    elif lr_scheduler_name == 'findlr':
        max_steps = kwargs['findlr_max_steps']
        lr_scheduler = FindLR(optimizer, max_steps)
    elif lr_scheduler_name == 'noam':
        lr_scheduler = NoamLR(optimizer, warmup_steps=warmup_steps)
    else:
        raise NotImplementedError("unknown lr_scheduler " + lr_scheduler_name)

    if warmup_steps > 0 and lr_scheduler_name != 'noam':
        lr_scheduler = WarmUpLR(lr_scheduler, warmup_steps, eta_min)

    lr_scheduler.name = lr_scheduler_name
    return lr_scheduler
