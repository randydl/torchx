import torch
import torch.nn.functional as F
import torchmetrics.functional as M


__all__ = ['accuracy']


def accuracy(outputs, targets):
    outputs = F.softmax(outputs, 1).argmax(1)
    return M.accuracy(outputs, targets)
