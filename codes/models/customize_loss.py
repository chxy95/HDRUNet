import torch
import torch.nn as nn

class tanh_L1Loss(nn.Module):
    def __init__(self):
        super(tanh_L1Loss, self).__init__()
    def forward(self, x, y):
        loss = torch.mean(torch.abs(torch.tanh(x) - torch.tanh(y)))
        return loss

class tanh_L2Loss(nn.Module):
    def __init__(self):
        super(tanh_L2Loss, self).__init__()
    def forward(self, x, y):
        loss = torch.mean(torch.pow((torch.tanh(x) - torch.tanh(y)), 2))
        return loss