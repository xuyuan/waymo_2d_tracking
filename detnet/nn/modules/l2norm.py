import torch
import torch.nn as nn
import torch.nn.init as init
from ..basenet import get_num_of_channels


class L2Norm(nn.Module):
    def __init__(self,in_channels, scale):
        super(L2Norm,self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


def l2norm(scale):
    return lambda last_layer: L2Norm(get_num_of_channels(last_layer), scale)