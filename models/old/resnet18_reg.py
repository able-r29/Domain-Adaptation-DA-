from typing import List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Module

import params
import utils
from .resnet18_mod import pretrained_resnet18
import squib.updaters.updater as squib_updater



def format_params():
    fmt        = '{}_{}_dr{}_tr{}_dim{}_a{}_b{}_c{}_d{}_p{}_l{}_norm{}_dL1{}_fix2'

    name       = params.model.name
    block      = params.model.resnet18_mod.block
    dr         = params.model.resnet18_mod.dr
    usetrained = params.model.use_trained
    n_dim      = params.model.triplet.n_dim
    alpha      = params.model.triplet.alpha
    beta       = params.model.triplet.beta
    gamma      = params.model.regulation.gamma
    delta      = params.model.triplet.delta
    pos        = params.model.regulation.pos
    level      = params.model.regulation.level
    norm       = params.model.triplet.norm
    difL1      = params.model.triplet.difL1

    dst = fmt.format(name, block, dr, usetrained, n_dim, alpha, beta, gamma, delta, pos, level, norm, difL1)
    return dst



class Model(Module):
    def __init__(self):
        super(Model, self).__init__()

        print('resnet18_triplet is used')

        n_class = params.dataset.n_class
        n_dim   = params.model.triplet.n_dim

        self.base = pretrained_resnet18()

        self.pos = params.model.regulation.pos
        n_ch = [64, 128, 256, 512]
        if n_dim != n_ch[self.pos]:
            self.fc1  = nn.Linear(n_ch[self.pos], n_dim, bias=params.model.fc_bias)
        else:
            self.fc1 = lambda x: x
        self.fc2  = nn.Linear(512, n_class,   bias=params.model.fc_bias)


    def forward(self, a, p):
        first = [
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
        ]
        layers = [
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
            self.base.layer4,
        ]
        pool = lambda x: torch.flatten(self.base.avgpool(x), 1)

        h = torch.cat([a, p], dim=0)

        for i, l in enumerate(first):
            h = l(h)

        for i, l in enumerate(layers):
            h = l(h)
            if i == self.pos:
                m = pool(h)
        h = pool(h)

        z = self.fc1(m)
        y = self.fc2(h)

        if params.model.triplet.norm:
            z = z / ((torch.sum(z*z, dim=1, keepdim=True)+1e-16)**0.5)

        ya, yp = torch.chunk(y, chunks=2, dim=0)
        za, zp = torch.chunk(z, chunks=2, dim=0)
        return ya, yp, za, zp



def semi_hard_loss(x, p, tx, tp):
    alpha = params.model.triplet.alpha

    dp = torch.sum((x - p)**2, dim=1)
    if params.model.triplet.difL1:
        dp = (dp+1e-16) ** 0.5

    t = torch.cat([tx, tp], dim=0)
    l = len(t)
    t = t.detach().cpu().numpy().astype(np.int64)
    m = np.stack([t for _ in range(l)])
    m = (m != m.T)[:l//2].astype(np.float32)

    c = torch.cat([x, p], dim=0)
    d = torch.stack([c for _ in range(l)])
    d = d - torch.transpose(d, 0, 1)
    d = torch.sum(d**2, dim=2)[:l//2]
    if params.model.triplet.difL1:
        d = (d+1e-16) ** 0.5

    dp_t = dp.detach().cpu().numpy().reshape(-1, 1)
    temp = d .detach().cpu().numpy()
    temp -= dp_t
    cand = np.logical_and(0<temp, temp<alpha)*np.random.randn(*temp.shape)
    mask = np.zeros_like(cand)
    mask[range(l//2),np.argmax(cand, axis=1)] = 1
    mask = torch.FloatTensor(mask*m).to(x.device, non_blocking=True)

    dn = torch.sum(d*mask, dim=1)
    mp = torch.sum(mask, dim=1)

    loss = torch.mean(F.relu(dp*mp-dn+alpha))

    return loss


def positive_loss(x, p):
    beta = params.model.triplet.beta
    if beta > 0:
        dp   = torch.sum((x - p)**2, dim=1)
        loss = torch.mean(F.relu(dp - beta))
    else:
        loss = 0
    return loss


def updater(model    :nn.Module,
            optimizer:optim.Optimizer=None,
            tag      :str=None,
            collecter:Callable[[torch.Tensor, torch.Tensor,
                                torch.Tensor, torch.Tensor], None]=None):
    cross_entropy_loss = nn.CrossEntropyLoss()
    
    def _update(a, p, ta, tp, la, lp, i=None):
        if optimizer is None:
            model.eval()
            with torch.no_grad():
                ya, yp, za, zp = model(a, p)
        else:
            model.train()
            ya, yp, za, zp = model(a, p)

        if collecter:
            collecter(ya, za, i)
        
        y = torch.cat([ya, yp], dim=0)
        t = torch.cat([ta, tp], dim=0)
        
        shl = semi_hard_loss(za, zp, la, lp)
        pos = positive_loss(za, zp)
        cel = cross_entropy_loss(y, t)
        acc = utils.TopN_accuracy(ya, ta, 1)


        loss = cel \
             + (shl + pos * params.model.triplet.delta) * params.model.regulation.gamma

        mets = {
            'shl':float(shl),
            'pos':float(pos),
            'cel':float(cel),
            'acc':float(acc)
        }

        return loss, mets
    
    upd = squib_updater.StanderdUpdater(loss_func=_update,
                                        optimizer=optimizer,
                                        tag      =tag)

    return upd


def metrices(tags:List[str]):
    loss = ['cel', 'shl', 'pos']
    top1 = ['acc']


    keys = [t+'/'+v for v in loss + top1 for t in tags]
    plot = {
        'loss.png':[t+'/'+v for v in loss for t in tags],
        'topn.png':[t+'/'+v for v in top1 for t in tags],
    }

    return keys, plot
