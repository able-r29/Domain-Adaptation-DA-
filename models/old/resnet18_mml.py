from typing import List, Callable

import numpy as np
import ignite
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Module
from ignite.metrics import Accuracy, Precision, Recall, Loss

import models.common as common
from .resnet18_base import pretrained_resnet18



class Model(Module):
    def __init__(self, n_class, fc_bias, n_meta, middle, **kargs):
        super(Model, self).__init__()

        self.n_class = n_class
        self.base    = pretrained_resnet18(**kargs)
        self.fc      = nn.Linear(middle, n_class+n_meta, bias=fc_bias)
        self.flatten = lambda x: torch.flatten(x, 1)


    def forward(self, x):
        x = x['img']
        layers = [
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.block11,
            self.base.block12,
            self.base.block21,
            self.base.block22,
            self.base.block31,
            self.base.block32,
            self.base.block41,
            self.base.block42,
            self.base.avgpool,
            self.flatten,
            self.fc
        ]

        for l in layers:
            x = l(x)
        
        y = x[:,:self.n_class]
        z = x[:,self.n_class:]
        z = torch.sigmoid(z)

        x = dict(pred=y, meta=z)
        return x


    def predict(self, *args, **kargs):
        result = self.forward(*args, **kargs)
        y = result['pred']
        z = result['meta']
        return y, z



def preprocess(batch, device, non_blocking):
    if len(batch) == 2:
        return common.common_preprocess(batch, device, non_blocking)

    x, y, m = batch
    x = x.to(device, non_blocking=non_blocking)
    y = y.to(device, non_blocking=non_blocking)
    m = m.to(device, non_blocking=non_blocking)

    x = dict(img=x)
    y = dict(label=y, meta=m)

    return x, y


def get_preprocess(**karg):
    return preprocess


def get_postprocess(**karg):
    return common.common_postprocess


def get_loss_func(alpha, **karg):
    cel = nn.CrossEntropyLoss()
    bce = nn.BCELoss()
    def _func(x, t):
        y = x['pred']
        z = x['meta']
        c = t['label']
        m = t['meta']
        l_cl = cel(y, c)
        l_ml = bce(z, m)
        return l_cl + alpha*l_ml
    return _func


def get_metrics(**karg):
    def _accuracy(output):
        x, t = output
        y = x['pred']
        t = t['label']
        return y, t
    def _round(output):
        x, t = output
        y = x['pred']
        t = t['label']
        y = torch.round(y)
        return y, t

    dst = {
        "accuracy"  : Accuracy (_accuracy),
        "precision" : Precision(_round),
        "recall"    : Recall   (_round),
        "loss"      : Loss     (get_loss_func(**karg))
    }
    return dst
