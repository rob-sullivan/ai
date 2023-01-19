"""
Copyright (c) 2017 Max deGroot, Ellis Brown
Released under the MIT license
https://github.com/amdegroot/ssd.pytorch
Updated by: me
"""
import torch
import torch.nn as nn
# from torch.autograd import Function
# from torch.autograd import Variable
import torch.nn.init as init
 
 
class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()
 
    def reset_parameters(self):
        # PyTorch 1.5.1
        # (fixes a warning)
        # init.constant(self.weight, self.gamma)
        init.constant_(self.weight, self.gamma)
 
    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        # PyTorch 1.5.1
        # (drastically increase the detection accuracy)
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out