import torch
import torch.nn as nn


class XYSigmoid(nn.Module):
    """
    apply sigmoid to x, y (first 2 index of last dim)
    """
    def forward(self, loc):
        loc[:, :, 0:2] = torch.sigmoid(loc[:, :, 0:2])
        return loc
