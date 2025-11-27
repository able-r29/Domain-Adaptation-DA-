from typing import List, Callable
from sklearn.metrics import roc_auc_score
import sys
sys.path.append('..')
import utils

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module, functional
from ignite.metrics import Accuracy, Loss, MetricsLambda, Metric

import models.common as common
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

        x = dict(ya=ya, yp=yp, za=za, zp=zp)
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



def semi_hard_loss(a, p, t, alpha):
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

    dp_t = dp.detach().cpu().numpy().reshape(-1, 1)
    temp = d .detach().cpu().numpy()
    temp -= dp_t
    cand = np.logical_and(0<temp, temp<alpha)*np.random.randn(*temp.shape)
    mask = np.zeros_like(cand)
    mask[range(l),np.argmax(cand, axis=1)] = 1
    mask = torch.FloatTensor(mask).to(a.device)

    dn = torch.sum(d*mask*m, dim=1)

    loss = torch.mean(functional.relu(dp-dn+alpha))

    return loss


def get_loss_func(alpha, gamma, **karg):
    cel = nn.CrossEntropyLoss()

    def _func(x, t):
        ya = x['ya']
        yp = x['yp']
        za = x['za']
        zp = x['zp']

        t = t['label']
        c = torch.tile(t, (2,))
        y = torch.cat([ya, yp], dim=0)

        l_cl = cel(y, c)
        l_mr = semi_hard_loss(za, zp, t, alpha)
        return l_cl + gamma*l_mr

    return _func


def get_metrics(alpha, **karg):
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
        l_mr = semi_hard_loss(a, p, t, alpha)
        return l_mr

    # カスタムメトリクスクラス
    class MacroSensitivityMetric(Metric):
        def __init__(self):
            super().__init__()
            self._predictions = []
            self._targets = []

        def reset(self):
            self._predictions = []
            self._targets = []

        def update(self, output):
            x, t = output
            y_pred = x['ya'].detach().cpu().numpy()
            y_true = t['label'].detach().cpu().numpy()
            
            self._predictions.append(y_pred)
            self._targets.append(y_true)

        def compute(self):
            if not self._predictions:
                return 0.0
            
            y_pred = np.concatenate(self._predictions, axis=0)
            y_true = np.concatenate(self._targets, axis=0)
            
            return utils.macro_sensitivity(y_pred, y_true, n_classes=2)

    class AUCMetric(Metric):
        def __init__(self):
            super().__init__()
            self._predictions = []
            self._targets = []

        def reset(self):
            self._predictions = []
            self._targets = []

        def update(self, output):
            x, t = output
            y_pred = torch.softmax(x['ya'], dim=1).detach().cpu().numpy()
            y_true = t['label'].detach().cpu().numpy()
            
            self._predictions.append(y_pred)
            self._targets.append(y_true)

        def compute(self):
            if not self._predictions:
                return 0.5
            
            y_pred = np.concatenate(self._predictions, axis=0)
            y_true = np.concatenate(self._targets, axis=0)
            
            try:
                if y_pred.ndim > 1 and y_pred.shape[1] > 1:
                    y_pred_proba = y_pred[:, 1]
                else:
                    y_pred_proba = y_pred
                
                return roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                return 0.5

    dst = {
        "Accuracy"        : Accuracy(_accuracy),
        "CrossEntropy"    : Loss(_cel),
        "SemiHardLoss"    : Loss(_shl),
        "MacroSensitivity": MacroSensitivityMetric(),
        "AUC"             : AUCMetric()
    }
    return dst
