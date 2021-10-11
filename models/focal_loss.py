import torch
import torch.nn as nn
import torch.nn.functional as F

"""
https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
"""
class BCEFocalLosswithLogits(nn.Module):
    def __init__(self, gamma=2, alpha=0.75, eps=1e-6, reduction='sum'):
        super(BCEFocalLosswithLogits, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps

    def forward(self, input, target):
        input = F.sigmoid(input)
        loss = -self.alpha * target * torch.pow(1. - input, self.gamma) * torch.log(target + self.eps) - \
               (1. - self.alpha) * (1. - target) * torch.pow(target, self.gamma) * torch.log(1. - input + self.eps)
        return loss.sum()