import collections
from typing import List, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, functional
from ignite.metrics import Accuracy, Loss

import datasets.dataset_meta_triplet as ds
import models.common as common
import utils
from .resnet18_base import pretrained_resnet18



class Model(Module):
    def __init__(self, n_class, fc_bias, n_meta, n_dim, middle, norm, **kargs):
        super(Model, self).__init__()

        self.n_class = n_class
        self.norm    = norm
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
        if self.norm:
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

    x, y, m = batch
    x = x.to(device, non_blocking=non_blocking)
    y = y.to(device, non_blocking=non_blocking)
    m = m.to(device, non_blocking=non_blocking)

    x = dict(img=x, meta=m)
    y = dict(label=y)

    return x, y


def get_preprocess(**karg):
    return preprocess


def get_postprocess(**karg):
    return common.common_postprocess


def kld(p, q):
    return torch.sum(torch.exp(p) * (p - q), dim=1)


def class_accuracy(path_fold, meta_use):
    def ratio(vs):
        vs  = list(map(lambda x:''.join(map(str, x)), vs.astype(np.int32)))
        dst = collections.Counter(vs)
        return dst

    def probability(mA, mB):
        key = set(mA.keys()) | set(mB.keys())

        mA = {k:(mA[k] if k in mA else 0) for k in key}
        mB = {k:(mB[k] if k in mB else 0) for k in key}

        nA = sum(mA.values())
        nB = sum(mB.values())
        n  = nA + nB

        pA = sum((mA[k]*mA[k])/(mA[k]+mB[k]) for k in key)
        pB = sum((mB[k]*mB[k])/(mA[k]+mB[k]) for k in key)
        return (pA + pB) / n

    ms = utils.load_json(path_fold)
    ms = sum(ms, start=[])
    ts = [m['LABEL'] for m in ms]
    ts = np.array(ts)
    vs = np.array([ds.meta_preprocess(m, meta_use) for m in ms])

    rs = [ratio(vs[ts==i]) for i in range(59)]

    mat = np.zeros((59, 59), dtype=np.float32)
    for i, row in enumerate(rs):
        for j, col in enumerate(rs):
            mat[i,j] = probability(row, col)
    mat = torch.FloatTensor(mat)
    return mat

    

def ranking(z1, t1, mat, gamma):
    z2 = torch.roll(z1, shifts=1, dims=0)
    t2 = torch.roll(t1, shifts=1, dims=0)
    z3 = torch.roll(z1, shifts=2, dims=0)
    t3 = torch.roll(t1, shifts=2, dims=0)

    d12 = torch.sum((z1 - z2)**2, dim=1)
    d13 = torch.sum((z1 - z3)**2, dim=1)
    p12 = mat[t1, t2]
    p13 = mat[t1, t3]

    dp = p12 - p13
    dz = d12 - d13
    ls = functional.relu(dz * torch.sign(dp)+gamma)
    return torch.mean(ls)


def binary(z1, t1, mat, *_):
    z2 = torch.roll(z1, shifts=1, dims=0)
    t2 = torch.roll(t1, shifts=1, dims=0)
    z3 = torch.roll(z1, shifts=2, dims=0)
    t3 = torch.roll(t1, shifts=2, dims=0)

    d12 = torch.sum((z1 - z2)**2, dim=1)
    d13 = torch.sum((z1 - z3)**2, dim=1)
    p12 = mat[t1, t2]
    p13 = mat[t1, t3]

    dp = p13 - p12
    t  = (torch.sign(dp)+1)*0.5
    dz = torch.sigmoid(d12 - d13)
    l  = functional.binary_cross_entropy(dz, t)
    return l


def get_loss_func(alpha, gamma, rank_type, path_fold, meta_use, device, **karg):
    cel = nn.CrossEntropyLoss()

    cmat = class_accuracy(path_fold, meta_use)
    cmat = cmat.to(device)

    func = {
        'ranking':ranking,
        'binary' :binary
    }[rank_type]

    def _func(x, t):
        y = x['pred']
        z = x['feat']
        c = t['label']

        l_cl = cel(y, c)
        l_mr = func(z, c, cmat, alpha)
        return l_cl + gamma*l_mr

    return _func


def get_metrics(alpha, rank_type, path_fold, meta_use, device, **karg):
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

    cmat = class_accuracy(path_fold, meta_use)
    cmat = cmat.to(device)
    def _rank(x, t):
        z = x['feat']
        c = t['label']

        func = {
            'ranking':ranking,
            'binary' :binary
        }[rank_type]
        l_mr = func(z, c, cmat, alpha)
        return l_mr

    dst = {
        "Accuracy"     : Accuracy (_accuracy),
        "CrossEntropy" : Loss     (_cel),
        "Ranking"      : Loss     (_rank)
    }
    return dst
