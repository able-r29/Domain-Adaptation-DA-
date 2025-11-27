from typing import List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Module

import params
import utils
import squib.updaters.updater as squib_updater
from .resnet18_mod import pretrained_resnet18
from .resnet18_reg import semi_hard_loss, positive_loss




def format_params():
    fmt        = '{}{}_dr{}_tr{}_dim{}_a{}_b{}_c{}_d{}_l{}_norm{}_difL1{}_fix2'

    name       = params.model.name
    dr         = params.model.resnet18_mod.dr
    usetrained = params.model.use_trained
    n_dim      = params.model.triplet.n_dim
    alpha      = params.model.triplet.alpha
    beta       = params.model.triplet.beta
    gamma      = params.model.regulation.gamma
    delta      = params.model.triplet.delta
    level      = params.model.regulation.level
    norm       = params.model.triplet.norm
    difL1      = params.model.triplet.difL1
    use_fc     = params.model.regulation_middle.use_fc

    dst = fmt.format(name, use_fc, dr, usetrained, n_dim, alpha, beta, gamma, delta, level, norm, difL1)
    return dst



class Model(Module):
    def __init__(self):
        super(Model, self).__init__()

        print('resnet18_triplet is used')

        n_class = params.dataset.n_class
        n_dim   = params.model.triplet.n_dim
        use_fc  = params.model.regulation_middle.use_fc

        self.base = pretrained_resnet18()
        self.fc1  = nn.Linear(512, n_class, bias=params.model.fc_bias)
        if use_fc:
            self.fcm = nn.ModuleList([
                    nn.Linear(f, n_dim) for f in [64,128,256,512]
                ])
        else:
            self.fcm = [lambda x: x] * 4


    def forward(self, a, p):
        layers = [
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
        ]
        blocks = [
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
            self.base.layer4,
        ]

        h = torch.cat([a, p], dim=0)

        for l in layers:
            h = l(h)
        
        ms = []
        for b, f in zip(blocks, self.fcm):
            h = b(h)
            ms.append(f(torch.flatten(self.base.avgpool(h), 1)))
        
        y = self.fc1(torch.flatten(self.base.avgpool(h), 1))

        zs = [m / ((torch.sum(m*m, dim=1, keepdim=True)+1e-16)**0.5) for m in ms]

        ya, yp = torch.chunk(y, chunks=2, dim=0)
        za, zp = zip(*[torch.chunk(z, chunks=2, dim=0) for z in zs])
        return ya, yp, za, zp



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
                ya, yp, zsa, zsp = model(a, p)
        else:
            model.train()
            ya, yp, zsa, zsp = model(a, p)

        if collecter:
            collecter(*([ya]+ list(zsa) + [i]))
        
        y = torch.cat([ya, yp], dim=0)
        t = torch.cat([ta, tp], dim=0)
        
        shl = sum([semi_hard_loss(za, zp, la, lp) for za, zp in zip(zsa, zsp)])/len(zsa)
        pos = sum([positive_loss(za, zp)          for za, zp in zip(zsa, zsp)])/len(zsa)
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
