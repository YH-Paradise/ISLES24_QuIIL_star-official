from torch import nn
import torch
from utils.common.loss_functions import DiceBCEFocalLoss


def optimizer_fc(model, init_lr, optim_name='adam', tmax_len=20):
    loss_fn = DiceBCEFocalLoss()

    if optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tmax_len, eta_min=1e-9)

    return loss_fn, optimizer, scheduler


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
