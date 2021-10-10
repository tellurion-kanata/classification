import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
"""
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):
        input_sigmoid = input.sigmoid()
        pt = (1 - input_sigmoid) * target + input_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none') * focal_weight
        loss = weight

