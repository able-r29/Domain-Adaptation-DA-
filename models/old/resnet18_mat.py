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
    def __init__(self, n_class, fc_bias, n_meta, d_att, middle, **kargs):
        super(Model, self).__init__()

        self.n_class     = n_class
        self.d_att       = d_att
        self.n_meta      = n_meta

        self.base = pretrained_resnet18(**kargs)
        self.q    = nn.Sequential(
            nn.Conv2d(512, middle, 1),
            nn.ReLU(),
            nn.Conv2d(middle, 2*d_att*n_meta, 1),
            nn.ReLU(),
        )
        self.k = nn.ModuleList([
            nn.Conv2d(128, d_att*n_meta, 1),
            nn.Conv2d(256, d_att*n_meta, 1),
        ])
        self.fc = nn.Sequential(
            nn.Linear(512, middle),
            nn.ReLU(),
            nn.Linear(middle, n_class, bias=fc_bias)
        )
    

    def attention(self, x, k, q, m):
        k = k(x)

        bch, _, hei, wid = k.shape
        k = k.reshape(bch*self.n_meta, self.d_att, hei*wid)
        k = k / (torch.sum(k*k, dim=1, keepdim=True) + 1e-8)**0.5

        l = torch.bmm(q, k)                     # (batch*n_meta, 1, hei*wid)
        a = torch.softmax(l, dim=2)             # (batch*n_meta, 1, hei*wid)
        a = a.reshape(bch, self.n_meta, hei, wid)
        s = torch.sum(a*m, dim=1, keepdim=True) + 1
        h = s * x

        return h, a



    def forward(self, x, requied_att=False):
        x, m = x
        m = m.reshape((m.shape[0], m.shape[1], 1, 1))

        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        h = x
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        x = self.base.avgpool(x)

        q  = self.q(x)
        q  = q.reshape((q.shape[0]*self.n_meta, 2, self.d_att))
        q  = q / (torch.sum(q*q, dim=2, keepdim=True) + 1e-8)**0.5
        qs = torch.chunk(q, chunks=2, dim=1)
        x  = h

        x, a2 = self.attention(x, self.k[0], qs[0], m)
        x     = self.base.layer3(x)
        x, a3 = self.attention(x, self.k[1], qs[1], m)
        x     = self.base.layer4(x)

        x = self.base.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if requied_att:
            a2 = torch.clone(a2*255).to(torch.uint8)
            a3 = torch.clone(a3*255).to(torch.uint8)
            return x, a2, a3
        return x


    def predict(self, *args, **kargs):
        return self.forward(requied_att=True, *args, **kargs)



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
