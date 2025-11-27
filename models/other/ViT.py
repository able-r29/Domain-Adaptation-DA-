import ignite
import torch
import torch.nn as nn
import models.common as common
from torch.nn import Module
import timm
import pprint



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)



def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, activation=None,
                 dr=0, mode='basic', **kargs):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        bnplanes = {
            'basic':planes,
            'wd'   :inplanes
        }[mode]

        self.conv1      = conv3x3(inplanes, planes, stride)
        self.bn1        = nn.BatchNorm2d(bnplanes) if norm_layer is None else norm_layer(planes)
        self.relu       = nn.ReLU(inplace=True)  if activation is None else activation()
        self.conv2      = conv3x3(planes, planes)
        self.bn2        = norm_layer(planes)
        self.downsample = downsample
        self.stride     = stride
        self.dr         = nn.Dropout2d(dr) if dr > 0 else None
        self.mode       = mode


    def forward_basic(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1  (out)
        out = self.relu (out)

        if self.dr is not None:
            out = self.dr(out)

        out = self.conv2(out)
        out = self.bn2  (out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


    def forward_wd(self, x):
        identity = x

        out = self.bn1  (x)
        out = self.relu (out)
        out = self.conv1(out)

        out = self.bn2(out)
        if self.dr is not None:
            out = self.dr(out)
        out = self.relu (out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


    def forward(self, x):
        func = {
            'basic':self.forward_basic,
            'wd'   :self.forward_wd
        }[self.mode]
        return func(x)



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, activation=None, **kargs):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        if activation is None:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = activation()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], **kargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], **kargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], **kargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], **kargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, **kargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, **kargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, **kargs))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def pretrained_resnet18(trained=None, **kargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kargs)
    if trained is not None:
        model.load_state_dict(torch.load(trained, map_location='cpu'))
    return model


class Model(Module):
    def __init__(self, n_class, fc_bias, **kargs):
        super(Model, self).__init__()
        print('resnet18_mod is used')

        self.n_class = n_class
        # self.base    = pretrained_resnet18(**kargs)
        # self.base = timm.create_model('vit_small_patch16_224_in21k', pretrained=True, num_classes=n_class)
        # self.base = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=n_class)
        # self.base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=n_class)
        self.base = timm.create_model('vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k', pretrained=True, num_classes=n_class)
        model_names = timm.list_models(pretrained=True)
        # pprint.pprint(model_names)
        # input()
        # self.fc      = nn.Linear(512, n_class, bias=fc_bias)
        # self.fc      = nn.Linear(75264, n_class, bias=fc_bias)
        self.fc      = nn.Linear(737280, n_class, bias=fc_bias)


    def forward(self, x):
        layers = [
            self.base.patch_embed,
            self.base.pos_drop,
            self.base.patch_drop,
            self.base.norm_pre,
            self.base.blocks,
            self.base.norm,
            self.base.fc_norm,
            self.base.head_drop,
            # self.base.head,
            lambda x: torch.flatten(x, 1),
            self.fc,
        ]
        for l in layers:
            x = l(x)

        return x


    def predict(self, *args, **kargs):
        return self.forward(*args, **kargs)


    def heat(self, x):
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
            self.fc,
        ]

        for l in layers:
            x = l(x)
            if l == self.base.layer4:
                h = torch.mean(x*x, dim=1)
                return h

        return x


    def cam(self, x, v_class):
        n_class = self.n_class

        layers = [
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
            self.base.layer4,
        ]

        for l in layers:
            x = l(x)

        ws = self.fc.weight.reshape(1, n_class, 512, 1, 1)
        vs = v_class.reshape(-1, n_class, 1, 1, 1)
        cw = torch.sum(ws * vs, dim=1, keepdim=False)

        cam = torch.sum(x * cw, dim=1)

        return cam


    def grad_cam_all(self, x:torch.Tensor, v_class:torch.Tensor):
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
            self.fc,
        ]

        maps  = []
        check = [
            self.base.conv1,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
            self.base.layer4,
        ]
        for l in layers:
            x = l(x)
            if l in check:
                x.retain_grad()
                maps.append(x)

        x.backward(v_class, retain_graph=True)

        alpha = [torch.mean(m.grad, dim=(2,3), keepdim=True) for m in maps]
        heats = [torch.clamp(torch.mean(m*a, dim=1), min=0) for m, a in zip(maps, alpha)]

        return heats


    def feature(self, x):
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
        ]

        for l in layers:
            x = l(x)

        y = self.fc(x)

        return x, y



def get_preprocess(**karg):
    return common.common_preprocess


def get_postprocess(**karg):
    return common.common_postprocess


def get_loss_func(loss_func, **karg):
    print('get_loss_funcget_loss_funcget_loss_funcget_loss_funcget_loss_funcget_loss_func')
    if loss_func == 'cross_entropy':
        print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
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
        "Accuracy" : ignite.metrics.Accuracy(),
        "Loss"     : ignite.metrics.Loss(get_loss_func(**karg))
    }
    return dst
