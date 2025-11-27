from typing import List, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, functional
from ignite.metrics import Accuracy, Loss

import models.common as common
from .resnet18_base import pretrained_resnet18



class Model(Module):
    def __init__(self, n_class, fc_bias, n_meta, n_dim, middle, **kargs):
        super(Model, self).__init__()

        self.n_class = n_class
        self.base    = pretrained_resnet18(**kargs)
        self.fc1     = nn.Linear(middle*2, n_class, bias=fc_bias)
        self.fc2     = nn.Linear(middle, n_dim, bias=fc_bias)
        self.fcm     = nn.Sequential(
            nn.Linear(n_meta, middle),
            nn.ReLU(),
            nn.Linear(middle, middle)
        )
        self.flatten = lambda x: torch.flatten(x, 1)


    def forward(self, x):
        m = x['meta']
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
        ]

        for l in layers:
            x = l(x)
        
        z = self.fc2(x)
        z = z / ((torch.sum(z*z, dim=1, keepdim=True)+1e-16) ** 0.5)

        m = self.fcm(m)
        c = torch.cat([x, m], dim=1)
        y = self.fc1(c)

        x = dict(pred=y, feat=z)
        return x


    def predict(self, *args, **kargs):
        result = self.forward(*args, **kargs)
        y = result['pred']
        z = result['feat']
        return y, z



def preprocess(batch, device, non_blocking):
    if len(batch) == 2:
        return common.common_preprocess(batch, device, non_blocking)

    x, y, m, p = batch
    x = x.to(device, non_blocking=non_blocking)
    y = y.to(device, non_blocking=non_blocking)
    m = m.to(device, non_blocking=non_blocking)
    p = p.to(device, non_blocking=non_blocking)

    x = dict(img=x, meta=m)
    y = dict(label=y, prob=p)

    return x, y


def get_preprocess(**karg):
    return preprocess


def get_postprocess(**karg):
    return common.common_postprocess


def kld(p, q):
    return torch.sum(torch.exp(p) * (p - q), dim=1)


def ranking(z1, p1, gamma):
    z2 = torch.roll(z1, shifts=1, dims=1)
    p2 = torch.roll(p1, shifts=1, dims=1)
    z3 = torch.roll(z1, shifts=2, dims=1)
    p3 = torch.roll(p1, shifts=2, dims=1)
    j1 = (kld(p1, p2) + kld(p2, p1)) * 0.5
    j2 = (kld(p1, p3) + kld(p3, p1)) * 0.5
    d1 = (torch.sum((z1 - z2)**2, dim=1) + 1e-16)**0.5
    d2 = (torch.sum((z1 - z3)**2, dim=1) + 1e-16)**0.5
    js = j1 - j2
    dz = d1 - d2
    ls = functional.relu(dz * torch.sign(js)+gamma)
    return torch.mean(ls)
    

def get_loss_func(alpha, gamma, **karg):
    cel = nn.CrossEntropyLoss()

    def _func(x, t):
        y = x['pred']
        z = x['feat']
        c = t['label']
        p = t['prob']
        l_cl = cel(y, c)
        l_mr = ranking(z, p, gamma)
        return l_cl + alpha*l_mr

    return _func


def get_metrics(gamma, **karg):
    def _accuracy(output):
        x, t = output
        y = x['pred']
        t = t['label']
        return y, t

    cel = nn.CrossEntropyLoss()
    def _cel(x, t):
        y = x['pred']
        t = t['label']
        l = cel(y, t)
        return l

    def _rank(x, t):
        z = x['feat']
        p = t['prob']
        l_mr = ranking(z, p, gamma)
        return l_mr

    dst = {
        "Accuracy"     : Accuracy (_accuracy),
        "CrossEntropy" : Loss     (_cel),
        "Ranking"      : Loss     (_rank)
    }
    return dst
