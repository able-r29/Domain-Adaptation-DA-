import collections

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
    def __init__(self, n_class, fc_bias, n_meta, n_dim, middle, **kargs):
        super(Model, self).__init__()

        self.n_class = n_class
        self.base    = pretrained_resnet18(**kargs)
        self.fc1     = nn.Linear(middle*2, n_class, bias=fc_bias)
        self.fc2     = nn.Linear(middle,   n_dim,   bias=fc_bias)
        self.fcm     = nn.Sequential(
            nn.Linear(n_meta, middle),
            nn.ReLU(),
            nn.Linear(middle, middle)
        )
        self.flatten = lambda x: torch.flatten(x, 1)


    def forward(self, x):
        a = x['anchor']
        p = x['positive']
        b = x['meta_a']
        q = x['meta_p']
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

        x = torch.cat([a, p], dim=0)

        for l in layers:
            x = l(x)

        z = self.fc2(x)
        z = z / ((torch.sum(z*z, dim=1, keepdim=True)+1e-16) ** 0.5)

        m = torch.cat([b, q], dim=0)
        m = self.fcm(m)
        c = torch.cat([x, m], dim=1)
        y = self.fc1(c)

        ya, yp = torch.chunk(y, chunks=2, dim=0)
        za, zp = torch.chunk(z, chunks=2, dim=0)

        x = dict(ya=ya, yp=yp, za=za, zp=zp, ma=b)
        return x


    def predict(self, x):
        m = x['meta_a']
        x = x['anchor']
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

        m = self.fcm(m)
        c = torch.cat([x, m], dim=1)
        y = self.fc1(c)
        z = self.fc2(x)
        z = z / ((torch.sum(z*z, dim=1, keepdim=True)+1e-16) ** 0.5)
        return y, z


    def feature(self, x):
        x = x['anchor']
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

        return x



def get_preprocess(**karg):
    def _preprocess(batch, device, non_blocking):
        a, p, b, q, t = batch
        a = a.to(device, non_blocking=non_blocking)
        p = p.to(device, non_blocking=non_blocking)
        b = b.to(device, non_blocking=non_blocking)
        q = q.to(device, non_blocking=non_blocking)
        t = t.to(device, non_blocking=non_blocking)

        x = dict(anchor=a, positive=p, meta_a=b, meta_p=q)
        y = dict(label=t)

        return x, y
    return _preprocess



def get_postprocess(**karg):
    return common.common_postprocess


def meta_right_ratio(ma, meat_right_ratio):
    meta_str = []
    for m in ma:
        m = m.cpu().numpy().copy()
        m = ''.join(map(str, (m.astype(np.int))))
        meta_str.append(m)

    meta_right_w = [meat_right_ratio[m] for m in meta_str]
    meta_w = 1 - torch.tensor(meta_right_w)
    return meta_w


def semi_hard_loss(a, p, t, alpha, meta_w):
    dp = torch.sum((a - p)**2, dim=1)

    t = t.detach().cpu().numpy().astype(np.int64)
    l = len(t)
    m = np.tile(t, (l*2, 2))
    m = (m != m.T)[:l].astype(np.float32)
    m = torch.FloatTensor(m).to(a.device)

    c = torch.cat([a, p], dim=0)
    d = torch.tile(c, (l*2, 1, 1))
    d = d - torch.transpose(d, 0, 1)
    d = torch.sum(d**2, dim=2)[:l]
    d = d * m

    dp_t = dp.detach().cpu().numpy().reshape(-1, 1)
    temp = d .detach().cpu().numpy()
    temp -= dp_t
    cand = np.logical_and(0<temp, temp<alpha)*np.random.rand(*temp.shape)
    inde = np.argmax(cand, axis=1)
    mask = np.zeros_like(cand)
    mask[range(l),inde] = 1
    mask = torch.FloatTensor(mask).to(a.device)

    weig = meta_w
    weig = torch.FloatTensor(weig).to(a.device)

    dn = torch.sum(d*mask, dim=1)
    ls = functional.relu(dp-dn+alpha)

    loss = torch.mean(ls*weig)

    return loss



def get_loss_func(alpha, gamma, path_fold, meta_use, n_class, meta_pred_weight_csv, **karg):
    cel = nn.CrossEntropyLoss()
    meat_right_ratio_dict = {}
    [meat_right_ratio_dict.update(utils.load_json(path)) for path in meta_pred_weight_csv]

    print('utils.load_json(meta_pred_weight_csv)')

    def _func(x, t):
        ya = x['ya']
        yp = x['yp']
        za = x['za']
        zp = x['zp']
        ma = x['ma']

        t = t['label']
        c = torch.tile(t, (2,))
        y = torch.cat([ya, yp], dim=0)

        meta_w = meta_right_ratio(ma, meat_right_ratio_dict)

        l_cl = cel(y, c)
        l_mr = semi_hard_loss(za, zp, t, alpha, meta_w)
        return l_cl + gamma*l_mr

    return _func


def get_metrics(alpha, path_fold, meta_use, n_class, meta_pred_weight_csv, **karg):
    meat_right_ratio_dict = {}
    [meat_right_ratio_dict.update(utils.load_json(path)) for path in meta_pred_weight_csv]
    print('utils.load_json(meta_pred_weight_csv)')

    def _accuracy(output):
        x, t = output
        y = x['ya']
        t = t['label']
        return y, t

    cel = nn.CrossEntropyLoss()
    def _cel(x, t):
        a = x['ya']
        p = x['yp']
        t = t['label']
        c = torch.tile(t, (2,))
        y = torch.cat([a, p], dim=0)
        l = cel(y, c)
        return l

    def _shl(x, t):
        a = x['za']
        p = x['zp']
        t = t['label']
        ma = x['ma']

        meta_w = meta_right_ratio(ma, meat_right_ratio_dict)
        l_mr = semi_hard_loss(a, p, t, alpha, meta_w)

        return l_mr

    dst = {
        "Accuracy"     : Accuracy (_accuracy),
        "CrossEntropy" : Loss     (_cel),
        "SemiHardLoss" : Loss     (_shl)
    }
    return dst
