from collections import OrderedDict
import ignite
import torch
import torch.nn as nn
import models.common as common
from torch.nn import Module



class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1, **kargs):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch_out)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch_out)

        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 1, stride, bias=False),
                nn.BatchNorm2d(ch_out),
            )
        else:
            self.downsample = lambda x: x


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1  (out)
        out = self.relu (out)

        out = self.conv2(out)
        out = self.bn2  (out)

        out += self.downsample(x)
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(self, num_classes=1000, block=BasicBlock, block_arg={}, **kargs):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.block11 = block(64,  64, 1,  **block_arg)
        self.block12 = block(64,  64, 1,  **block_arg)
        self.block21 = block(64,  128, 2, **block_arg)
        self.block22 = block(128, 128, 1, **block_arg)
        self.block31 = block(128, 256, 2, **block_arg)
        self.block32 = block(256, 256, 1, **block_arg)
        self.block41 = block(256, 512, 2, **block_arg)
        self.block42 = block(512, 512, 1, **block_arg)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block11(x)
        x = self.block12(x)
        x = self.block21(x)
        x = self.block22(x)
        x = self.block31(x)
        x = self.block32(x)
        x = self.block41(x)
        x = self.block42(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def feature(self, x):
        """特徴抽出メソッド（最終分類層前の512次元特徴量）"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.block11(x)
        x = self.block12(x)
        x = self.block21(x)
        x = self.block22(x)
        x = self.block31(x)
        x = self.block32(x)
        x = self.block41(x)
        x = self.block42(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # 最終分類層を通さずに512次元特徴量を返す
        return x


def pretrained_resnet18(trained=None, **kargs):
    model = ResNet(**kargs)

    if trained is not None:
        src   = torch.load(trained, map_location='cpu')
        rep   = OrderedDict()
        table = [
            ['layer1.0', 'block11'],
            ['layer1.1', 'block12'],
            ['layer2.0', 'block21'],
            ['layer2.1', 'block22'],
            ['layer3.0', 'block31'],
            ['layer3.1', 'block32'],
            ['layer4.0', 'block41'],
            ['layer4.1', 'block42'],
        ]
        for k, v in src.items():
            for t in table: k = k.replace(*t)
            rep[k] = v
        model.load_state_dict(rep)
    return model


class Model(Module):
    def __init__(self, n_class, fc_bias, **kargs):
        super(Model, self).__init__()
        print('resnet18_base is used')

        self.n_class = n_class
        self.base    = pretrained_resnet18(**kargs)
        self.fc      = nn.Linear(512, n_class, bias=fc_bias)


    def forward(self, x):
        # 最終分類まで行う
        features = self.base.feature(x)  # 512次元特徴
        logits = self.fc(features)       # 分類
        return logits

    def predict(self, *args, **kargs):
        return self.forward(*args, **kargs)

    def feature(self, x):
        """特徴抽出専用メソッド（DANNで使用）"""
        return self.base.feature(x)  # 512次元特徴量を返す


def get_preprocess(**karg):
    return common.common_preprocess


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
        "Accuracy"     : ignite.metrics.Accuracy(),
        "CrossEntropy" : ignite.metrics.Loss(get_loss_func(**karg))
    }
    return dst


if __name__ == '__main__':
    import models.resnet18_mod as mod
    res1 = mod.pretrained_resnet18(r'T:\masaya\nsdd\metas\models\resnet18.pth')
    res2 = pretrained_resnet18(r'T:\masaya\nsdd\metas\models\resnet18.pth')

    with torch.inference_mode():
        x  = torch.randn((4, 3, 448, 448))
        y1 = res1(x)
        y2 = res2(x)
        d  = torch.mean(torch.abs(y1-y2))

    print(d)
