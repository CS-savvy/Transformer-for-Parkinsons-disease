import torch.nn as nn
import torch
import torch.nn.functional as F


class Losses:

    def __init__(self):
        self.losses = {'Focal': FocalLoss,
                       'BinaryCrossEntropy': nn.BCEWithLogitsLoss,
                       'CrossEntrory': nn.CrossEntropyLoss}

    def get_criterion(self, name):
        return self.losses[name]


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        f_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(f_loss)
        else:
            return f_loss

