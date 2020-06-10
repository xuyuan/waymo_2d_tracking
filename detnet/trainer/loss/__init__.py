from torch import nn


class FunctionLoss(nn.Module):
    def __init__(self, func_loss):
        super().__init__()
        self.func_loss = func_loss

    def forward(self, *input):
        return self.func_loss(*input)


class TargetedLoss(nn.Module):
    def __init__(self, criterion, target_name):
        super().__init__()
        self.criterion = criterion
        self.target_name = target_name

    def forward(self, input, target):
        target = target[self.target_name]
        return self.criterion(input, target)


class CollectionLoss(nn.Module):
    def __init__(self, criterions):
        super().__init__()
        criterions = {k.replace('.', '-'): self._convert_function_to_module(m) for k, m in criterions.items()}
        self.criterions = nn.ModuleDict(criterions)

    def forward(self, *input):
        return {n: c(*input) for n, c in self.criterions.items()}

    def _convert_function_to_module(self, criterion):
        if isinstance(criterion, nn.Module):
            return criterion
        return FunctionLoss(criterion)


class WeightedLoss(nn.Module):
    def __init__(self, criterion, weight):
        super().__init__()
        self.criterion = criterion
        self.weight = weight

    def forward(self, *input):
        l = self.criterion(*input)
        return l * self.weight

    def extra_repr(self):
        return f'(weight): {self.weight}'
