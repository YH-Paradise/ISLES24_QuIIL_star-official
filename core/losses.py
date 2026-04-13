import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCEFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, size_average=True):
        super(DiceBCEFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, smooth=1e-2):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        # targets = targets.view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        bceloss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_for_focal = F.binary_cross_entropy(inputs, targets, reduction='mean')
        pt = torch.exp(-BCE_for_focal)

        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_for_focal

        # return bceloss + dice_loss + F_loss
        return dice_loss + F_loss
