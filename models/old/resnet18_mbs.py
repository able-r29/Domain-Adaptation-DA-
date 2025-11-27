from typing import List, Callable

import numpy as np
import ignite
import torch
import torch.nn as nn
from torch.nn import Module

import models.common as common
from .resnet18_base import pretrained_resnet18



class Model(Module):
    def __init__(self, n_class, fc_bias, n_meta, middle, trained1, trained2, **kargs):
        super(Model, self).__init__()

        self.n_class = n_class
        self.base1   = pretrained_resnet18(trained1, **kargs)
        self.base2   = pretrained_resnet18(trained2, **kargs)
        self.fc      = nn.Linear(middle, n_class, bias=fc_bias)
        self.flatten = lambda x: torch.flatten(x, 1)

        self.meta   = nn.Sequential(
            nn.Linear(n_meta, middle),
            nn.ReLU(),
            nn.Linear(middle, middle),
            nn.ReLU(),
            nn.Linear(middle, 2*8)
        )
    

    def forward(self, x, request_att=False):
        x, m = x

        vs = self.meta(m).reshape(x.shape[0], 8, 2, 1, 1, 1)
        vs = torch.softmax(vs, dim=2)

        x = self.base1.conv1(x)
        x = self.base1.bn1(x)
        x = self.base1.relu(x)
        x = self.base1.maxpool(x)

        bs1 = [self.base1.block11, self.base1.block12,
               self.base1.block21, self.base1.block22,
               self.base1.block31, self.base1.block32,
               self.base1.block41, self.base1.block42]
        bs2 = [self.base2.block11, self.base2.block12,
               self.base2.block21, self.base2.block22,
               self.base2.block31, self.base2.block32,
               self.base2.block41, self.base2.block42]
        for i, (b1, b2) in enumerate(zip(bs1, bs2)):
            x1 = b1(x)
            x2 = b2(x)
            a1 = vs[:,i,0]
            a2 = vs[:,i,1]
            x  = x1*a1 + x2*a2

        x = self.base1.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        if request_att:
            return x, vs

        return x


    def predict(self, *args, **kargs):
        return self.forward(request_att=True, *args, **kargs)



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
        "accuracy" : ignite.metrics.Accuracy(),
        "loss"     : ignite.metrics.Loss(get_loss_func(**karg))
    }
    return dst
