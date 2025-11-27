from typing import List, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module

import params
import utils
import squib.updaters.updater as upd
from . import resnet18_mod
from .common import MultilossMetrices



def metrices(tags:List[str]):
    return MultilossMetrices(2, tags)



def format_params():
    fmt        = '{}_dr{}_a{}_tr{}'

    name       = params.model.name
    dr         = params.model.resnet18_mod.dr
    alpah      = params.model.resnet18_spr.alpha
    usetrained = params.model.use_trained

    dst = fmt.format(name, dr, alpah, usetrained)
    return dst



class Model(Module):
    def __init__(self):
        super(Model, self).__init__()

        print('resnet18_mod is used')

        n_class = params.dataset.n_class

        self.base = resnet18_mod.pretrained_resnet18()
        self.cnv  = nn.Conv2d(512, n_class, kernel_size=1, bias=params.model.fc_bias)


    def forward(self, x):
        layers = [
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
            self.base.layer4,
            self.cnv,
            self.base.avgpool,
            lambda x: torch.flatten(x, 1),
        ]

        batch, sim, ch, hei, wid = map(int, x.shape)
        x = x.reshape(batch*sim, ch, hei, wid)

        for l in layers:
            if l == self.base.avgpool:
                e = x
            x = l(x)

        return x, e
    
    def cam(self, x, v_class):
        n_class = params.dataset.n_class

        layers = [
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
            self.base.layer4,
            self.cnv,
        ]

        batch, sim, ch, hei, wid = map(int, x.shape)
        x = x.reshape(batch*sim, ch, hei, wid)

        for l in layers:
            x = l(x)
        
        vs = v_class.reshape(batch*sim, n_class, 1, 1)

        cam = torch.sum(x * vs, dim=1)

        return cam


def updater(model    :nn.Module,
            optimizer:optim.Optimizer=None,
            tag      :str=None,
            collecter:Callable[[torch.Tensor, torch.Tensor,
                                torch.Tensor, torch.Tensor], None]=None):
    func_cel = nn.CrossEntropyLoss()
    func_ele = lambda x: torch.mean(x*x)
    
    def _update(x, t, i=None):
        if optimizer is None:
            model.eval()
        else:
            model.train()

        batch, sim, _, _, _ = map(int, x.shape)
        t = t.reshape(batch*sim)
        if i is not None:
            i = i.reshape(batch*sim)

        y, e = model(x)

        if collecter:
            collecter(x, y, t, i)

        loss_cel = func_cel(y, t)
        loss_ele = func_ele(e)
        sum_loss = loss_cel + loss_ele * params.model.resnet18_spr.alpha

        top1 = utils.TopN_accuracy(y, t, 1)

        mets = {
            'loss1': loss_cel.item(),
            'loss2': loss_ele.item(),
            'top1':top1,
        }

        return sum_loss, mets
    
    u = upd.StanderdUpdater(loss_func=_update, optimizer=optimizer, tag=tag)

    return u
