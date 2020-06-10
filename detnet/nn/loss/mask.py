
from torch import nn as nn
from .focal import FocalLoss
from .lovasz import LovaszLoss


class MaskLoss(nn.Module):
    def __init__(self, loss_type, **kwargs):
        super().__init__()
        if loss_type == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss(**kwargs)
        elif loss_type == 'Focal':
            self.criterion = FocalLoss(2, **kwargs)
        elif loss_type == 'Lovasz':
            self.criterion = LovaszLoss()
        else:
            raise NotImplementedError(loss_type)

    def forward(self, logits, targets):
        """
        Parameters
        ----------
        logits: torch.autograd.Variable
        targets: torch.autograd.Variable

        Returns
        -------
        loss
        """
        if logits is None:
            return None
        logits = logits.float()
        targets = targets.long()
        return self.criterion(logits, targets)


class ElevationLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        #self.criterion = nn.MSELoss(**kwargs)
        self.criterion = nn.SmoothL1Loss(**kwargs)
        #self.criterion = DiceLoss()

    def forward(self, input, target):
        if input is None:
            return None
        return self.criterion(input, target)