import torch
import torch.nn as nn
from torch.nn import Module
import torchvision

import params
from .common import create_train_engine, create_eval_engine, get_metrics_name


def format_params():
    fmt        = 'model_{}_tr{}'

    name       = params.model.name
    usetrained = params.model.use_trained

    dst = fmt.format(name, usetrained)
    return dst


def pretrained_googlenet():
    path  = params.model.trained_path.googlenet
    model = torchvision.models.GoogLeNet(aux_logits=False)
    if params.model.use_trained:
        model.load_state_dict(torch.load(path, map_location='cpu'))
        print('trained model is used')
    return model


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()

        print('googlenet is used')

        n_class = params.dataset.n_class

        self.base = pretrained_googlenet()
        self.fc   = nn.Linear(1024, n_class)

        self.lsmx = nn.LogSoftmax(dim=1)


    def forward(self, x):
        layers = [
            self.base.conv1,
            self.base.maxpool1,
            self.base.conv2,
            self.base.conv3,
            self.base.maxpool2,
            self.base.inception3a,
            self.base.inception3b,
            self.base.maxpool3,
            self.base.inception4a,
            self.base.inception4b,
            self.base.inception4c,
            self.base.inception4d,
            self.base.inception4e,
            self.base.maxpool4,
            self.base.inception5a,
            self.base.inception5b,
            self.base.avgpool,
            lambda x: torch.flatten(x, 1),
            self.base.dropout,
            self.fc,
            self.lsmx,
        ]

        for l in layers:
            x = l(x)
        
        return x


if __name__ == "__main__":
    path  = params.model.trained_path.googlenet
    model = torchvision.models.googlenet(pretrained=True)
    torch.save(model.state_dict(), path)
