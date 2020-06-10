import torch
import torch.nn as nn

class AdaptiveConcatPool2d(nn.Module):
    '''http://forums.fast.ai/t/what-is-the-distinct-usage-of-the-adaptiveconcatpool2d-layer/7600
    Sometimes Max value of the HxW feature map of the last layer works better than Avg and vice-versa.
    By adding both in the final layer, you are letting Neural Net choose what works without having to experiment yourself.
    '''
    def __init__(self, output_size=None):
        super().__init__()
        output_size = output_size or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(output_size)
        self.mp = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)