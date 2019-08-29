import torch
from torch.nn import functional as F

EPS = 1e-9


class EDL_Loss(torch.nn.Module):

    def __init__(self):
        super(EDL_Loss, self).__init__()

    def forward(self, true, pred):
        true = torch.clamp(true, min=EPS, max=1 - EPS)
        pred = torch.clamp(pred, min=EPS, max=1 - EPS)

        F.normalize(true, p=1, dim=1)
        F.normalize(pred, p=1, dim=1)

        beta = 0.7
        N = true.size()[0]

        # classification loss
        max_val, _ = torch.max(true, 1)
        max_val = max_val.view(max_val.shape[0], 1)

        label = (true == max_val).float()
        ce = -torch.sum(torch.mul(label, torch.log(pred))) / N

        # kl loss
        kl = torch.sum(torch.mul(true, torch.log(torch.div(true, pred)))) / N

        return beta * kl + (1 - beta) * ce