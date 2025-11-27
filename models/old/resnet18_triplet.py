from typing import List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Module

import params
from .resnet18_mod import pretrained_resnet18
import squib.updaters.updater as squib_updater



def format_params():
    fmt        = '{}_dr{}_tr{}_dim{}_a{}_norm{}_fix'

    name       = params.model.name
    dr         = params.model.resnet18_mod.dr
    usetrained = params.model.use_trained
    n_dim      = params.model.triplet.n_dim
    alpha      = params.model.triplet.alpha
    beta       = params.model.triplet.beta
    norm       = params.model.triplet.norm

    dst = fmt.format(name, dr, usetrained, n_dim, alpha, beta, norm)
    return dst



class Model(Module):
    def __init__(self):
        super(Model, self).__init__()

        print('resnet18_triplet is used')

        n_dim     = params.model.triplet.n_dim

        self.base = pretrained_resnet18()
        self.fc   = nn.Linear(512, n_dim, bias=params.model.fc_bias)


    def forward(self, x, n, p):
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
            self.fc
        ]

        h = torch.cat([x, n, p], dim=0)

        for l in layers:
            h = l(h)

        if params.model.triplet.norm:
            h = h / ((torch.sum(h*h, dim=1, keepdim=True)+1e-16)**0.5)

        x, n, p = torch.chunk(h, chunks=3, dim=0)
        return x, n, p



def updater(model    :nn.Module,
                   optimizer:optim.Optimizer=None,
                   tag      :str=None,
                   collecter:Callable[[torch.Tensor, torch.Tensor,
                                       torch.Tensor, torch.Tensor], None]=None):

    def loss_func(x, p, n, tx, tp, tn):
        x = torch.cat([ x,  p,  n], dim=0)
        t = torch.cat([tx, tp, tn], dim=0)
        t = t.detach().cpu().numpy().astype(np.int64)
        n = len(t)
        m = t.repeat(n).reshape(n, n)
        m = (m == m.T).reshape(n, n).astype(np.float32)
        m = torch.FloatTensor(m).to(x.device)

        d = torch.stack([x for _ in range(n)])
        d = d - torch.transpose(d, 0, 1)
        d = torch.sum(d * d, dim=2)

        dp = m * d
        dn = d + m * 4

        dp = torch.max(dp, dim=0)[0][:n//3]
        dn = torch.min(dn, dim=0)[0][:n//3]
        
        alpha = params.model.triplet.alpha
        loss  = torch.mean(torch.nn.functional.relu(dp-dn + alpha))
        return loss

    
    def semi_hard_loss(x, p, n, tx, tp, tn):
        alpha = params.model.triplet.alpha

        dp = torch.sum((x - p)**2, dim=1)

        t = torch.cat([tx, tp, tn], dim=0)
        l = len(t)
        t = t.detach().cpu().numpy().astype(np.int64)
        m = np.stack([t for _ in range(l)])
        m = (m != m.T)[:l//3].astype(np.float32)
        m = torch.FloatTensor(m).to(x.device)

        c = torch.cat([ x,  p,  n], dim=0)
        d = torch.stack([c for _ in range(l)])
        d = d - torch.transpose(d, 0, 1)
        d = torch.sum(d**2, dim=2)[:l//3]

        dp_t = dp.detach().cpu().numpy().reshape(-1, 1)
        temp = d .detach().cpu().numpy()
        temp -= dp_t
        cand = np.logical_and(0<temp, temp<alpha)*np.random.randn(*temp.shape)
        mask = np.zeros_like(cand)
        mask[range(l//3),np.argmax(cand, axis=1)] = 1
        mask = torch.FloatTensor(mask).to(x.device)

        dn = torch.sum(d*mask*m, dim=1)

        loss = torch.mean(F.relu(dp-dn+alpha))

        return loss

    def positive_loss(x, p):
        beta = params.model.triplet.beta
        if beta > 0:
            dp = torch.sum((x - p)**2, dim=1)
            loss  = torch.mean(F.relu(dp - beta))
        else:
            loss = 0
        return loss

    
    def _update(x, p, n, tx, tp, tn, i=None):
        if optimizer is None:
            model.eval()
            with torch.no_grad():
                y, p, n = model(x, p, n)
        else:
            model.train()
            y, p, n = model(x, p, n)

        if collecter:
            collecter(y, i)

        loss = semi_hard_loss(y, p, n, tx, tp, tn)# + \
            #    positive_loss(y, p)

        mets = {
            'loss':loss.item(),
        }

        return loss, mets
    
    upd = squib_updater.StanderdUpdater(loss_func=_update, optimizer=optimizer, tag=tag)

    return upd


def metrices(tags:List[str]):
    loss = ['loss']

    keys = [t+'/'+v for v in loss for t in tags]
    plot = {
        'loss.png':[t+'/'+v for v in loss for t in tags],
    }

    return keys, plot
