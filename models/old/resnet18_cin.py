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
    def __init__(self, n_class, fc_bias, n_meta, middle, apply_index, **kargs):
        super(Model, self).__init__()

        self.apply_index = apply_index

        base = pretrained_resnet18(**kargs)
        self.cb = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )
        self.blocks = nn.ModuleList(
            [
                base.layer1,
                base.layer2,
                base.layer3,
                base.layer4,
            ]
        )
        self.fc = nn.Sequential(
            base.avgpool,
            common.Lambda(lambda x: torch.flatten(x, 1)),
            nn.Linear(512, n_class, bias=fc_bias)
        ) 

        self.meta    = nn.Sequential(
            nn.Linear(n_meta, middle),
            nn.ReLU(),
            nn.Linear(middle, middle),
            nn.ReLU()
        )
        self.lins = nn.ModuleList(
            [
                nn.Linear(middle,  64*2),
                nn.Linear(middle, 128*2),
                nn.Linear(middle, 256*2),
                nn.Linear(middle, 512*2),
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.BatchNorm2d(64,  affine=False),
                nn.BatchNorm2d(128, affine=False),
                nn.BatchNorm2d(256, affine=False),
                nn.BatchNorm2d(512, affine=False),
            ]
        )
        self.scale = nn.ParameterList([
            nn.parameter.Parameter(torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32)),
            nn.parameter.Parameter(torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32)),
            nn.parameter.Parameter(torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32)),
            nn.parameter.Parameter(torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32)),
        ])
        self.softplus = nn.Softplus()


    def forward(self, x, request_ms=False):
        mss  = []
        x, m = x

        style = self.meta(m)

        x = self.cb(x)
        for i_block, (block, lin, scl, norm) in enumerate(zip(self.blocks, self.lins,
                                                              self.scale,  self.norms)):
            x = block(x)
            if i_block in self.apply_index:
                x  = norm(x)
                ms = lin(style).reshape(x.shape[0], 2, x.shape[1], 1, 1)
                ms = ms * scl
                mss.append(ms)
                m  = ms[:, 0]
                s  = self.softplus(ms[:, 1])
                x  = s*x + m

        x = self.fc(x)

        if request_ms:
            return x, mss
        return x
    

    def predict(self, x):
        x, mss = self.forward(x, True)
        return x, *mss



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


def get_loss_func(**karg):
    return common.common_loss_func(**karg)


def get_metrics(**karg):
    return common.common_metrics(**karg)
