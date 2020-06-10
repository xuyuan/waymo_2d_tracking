from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, ReduceLROnPlateau


class FindLR(_LRScheduler):
    """
    inspired by fast.ai @https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """
    def __init__(self, optimizer, max_steps, max_lr_scale=100):
        self.max_steps = max_steps
        self.max_lr_scale = max_lr_scale
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.max_lr_scale ** (self.last_epoch / (self.max_steps - 1)))
                for base_lr in self.base_lrs]


class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


class WarmUpLR(_LRScheduler):
    def __init__(self, lr_scheduler, warmup_steps, eta_min=0, last_epoch=-1):
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        self.optimizer = lr_scheduler.optimizer
        self._step_count = 0
        self.last_epoch = last_epoch

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in self.optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        self.base_lrs = list(map(lambda group: group['initial_lr'], self.optimizer.param_groups))
        self._update_lr()

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.eta_min + (base_lr - self.eta_min) * ((self.last_epoch + 1) / self.warmup_steps)
                    for base_lr in self.base_lrs]
        return [group['lr'] for group in self.optimizer.param_groups]

    def step(self, metrics=None):
        self.last_epoch += 1

        if isinstance(self.lr_scheduler, ReduceLROnPlateau):
            if metrics is not None:
                self.lr_scheduler.step(metrics)
        else:
            self.lr_scheduler.step()

        self._update_lr()

    def _update_lr(self):
        if self.last_epoch < self.warmup_steps:
            values = self.get_lr()
            for param_group, lr in zip(self.optimizer.param_groups, values):
                param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_scheduler')}
        state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        lr_scheduler = state_dict.pop('lr_scheduler')
        self.__dict__.update(state_dict)
        self.lr_scheduler.load_state_dict(lr_scheduler)
        self.last_epoch = self.lr_scheduler.last_epoch + self.warmup_steps

