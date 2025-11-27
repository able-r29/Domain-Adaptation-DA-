from typing import List, Callable

import numpy as np
import ignite
import torch
import torch.nn as nn
from torch.nn import Module
from torch.types import Number

import models.common as common
from .resnet18_base import pretrained_resnet18



class MetaSEBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1, **kargs):
        super(MetaSEBlock, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch_out)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch_out)

        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 1, stride, bias=False),
                nn.BatchNorm2d(ch_out),
            )
        else:
            self.downsample = lambda x: x


    def forward(self, x, v=None):
        out = self.conv1(x)
        out = self.bn1  (out)
        out = self.relu (out)

        out = self.conv2(out)
        out = self.bn2  (out)

        if v is not None:
            out = out * v

        out += self.downsample(x)
        out = self.relu(out)

        return out



class Model(Module):
    def __init__(self, n_class, fc_bias, n_meta, middle, apply_index, dr=None, **kargs):
        super(Model, self).__init__()

        self.n_class = n_class
        self.base    = pretrained_resnet18(block=MetaSEBlock, **kargs)
        self.fc      = nn.Linear(middle, n_class, bias=fc_bias)
        self.dr      = nn.Dropout(dr) if dr is not None else lambda x: x

        self.n_mids = [64, 64, 128, 128, 256, 256, 512, 512]
        self.index  = apply_index
        self.meta   = nn.Sequential(
            nn.Linear(n_meta, middle),
            nn.ReLU(),
            nn.Linear(middle, middle),
            nn.ReLU(),
            nn.Linear(middle, sum(self.n_mids)),
            nn.Sigmoid()
        )


    def forward(self, x, request_att=False):
        x, m = x

        vs = self.meta(m).reshape(x.shape[0], -1, 1, 1)
        cs = 0
        bs = [self.base.block11, self.base.block12,
              self.base.block21, self.base.block22,
              self.base.block31, self.base.block32,
              self.base.block41, self.base.block42]

        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        for i, (n, b) in enumerate(zip(self.n_mids, bs)):
            v   = vs[:,cs:cs+n] if i in self.index else None
            cs += n
            x   = b(x, v)

        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dr(x)
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
