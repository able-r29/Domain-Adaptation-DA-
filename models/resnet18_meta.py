from typing import List, Callable

import numpy as np
import ignite
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Module

import models.common as common
from .resnet18_mod import pretrained_resnet18



class Model(Module):
    def __init__(self, n_class, fc_bias, n_meta, middle, **kargs):
        super(Model, self).__init__()

        self.n_class = n_class
        self.base    = pretrained_resnet18(**kargs)
        if n_meta > 0:
            self.meta    = nn.Sequential(
                nn.Linear(n_meta, middle),
                nn.ReLU(),
                nn.Linear(middle, middle),
                nn.ReLU(),
            )
            init = middle
        else:
            init = 0
        # self.fc      = nn.Sequential(
            # nn.Linear(512+init, middle),
            # nn.ReLU(),
            # nn.Linear(middle,   n_class, bias=fc_bias)
        # )
        self.fc     = nn.Linear(middle*2, n_class, bias=fc_bias)


    def forward(self, x):
        x, m = x

        layers = [
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
            self.base.layer4,
            self.base.avgpool,
            lambda x: torch.flatten(x, 1),
        ]

        for l in layers:
            x = l(x)
        if m is not None:
            m = self.meta(m)
            x = torch.cat([x, m], dim=1)
        x = self.fc(x)

        return x


    def predict(self, *args, **kargs):
        return self.forward(*args, **kargs)


    def feature(self, x):
        if isinstance(x, tuple):
            x, m = x
        else:
            m = None

        layers = [
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
            self.base.layer4,
            self.base.avgpool,
            lambda x: torch.flatten(x, 1),
        ]

        for l in layers:
            x = l(x)

        return x



def preprocess(batch, device, non_blocking):
    if len(batch) == 2:
        return common.common_preprocess(batch, device, non_blocking)

    x, y, m = batch
    x = x.to(device, non_blocking=non_blocking)
    y = y.to(device, non_blocking=non_blocking)
    m = m.to(device, non_blocking=non_blocking)

    return (x, m), y


def get_preprocess(**karg):
    return preprocess


def get_postprocess(**karg):
    return common.common_postprocess


def get_loss_func(loss_func, **karg):
    if loss_func == 'cross_entropy':
        return nn.CrossEntropyLoss()
    if loss_func == 'focal_loss':
        gamma = karg['gamma']
        def _func(x, t):
            l = torch.log_softmax(x, dim=1)
            s = torch.exp(l)
            t = torch.zeros(x.shape, dtype=torch.float32, device=x.device).scatter_(1, t.reshape(-1, 1), 1)
            s = torch.sum(     t * s, dim=1)
            h = torch.sum(-1 * t * l, dim=1)
            f = (1-s).pow(gamma) * h
            return torch.mean(f)
        return _func


def get_metrics(**karg):
    dst = {
        "Accuracy" : ignite.metrics.Accuracy(),
        "loss"     : ignite.metrics.Loss(get_loss_func(**karg))
    }
    return dst
