import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.nn import Module
import ignite.engine as engine
import ignite.metrics as metrics

import params
import utils
from .common import Lambda


def format_params():
    fmt        = 'model_{}_tr{}_fcbias{}_sim{}_dk{}_batchnorm{}_attnorm{}_vecnorm{}_att{}'

    name       = params.model.name
    usetrained = params.model.use_trained
    fc_bias    = params.model.fc_bias
    n_sim      = params.model.att_trd.n_sim
    dk         = params.model.att_trd.dk
    batchnorm  = params.model.att_trd.batchnorm
    attnorm    = params.model.att_trd.attnorm
    vecnorm    = params.model.att_trd.vecnorm
    att_func   = params.model.att_trd.att_func

    dst = fmt.format(name, usetrained, fc_bias, n_sim, dk, batchnorm, attnorm, vecnorm, att_func)
    return dst


class FocusAttention(Module):
    def __init__(self, in_ch):
        super(FocusAttention, self).__init__()
        n_class = params.dataset.n_class

        self.focus = nn.Sequential(
            nn.Conv2d(in_ch, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.loss = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            Lambda(lambda x: torch.flatten(x, 1)),
            nn.Linear(in_ch, n_class, bias=False),
            nn.LogSoftmax(dim=1),
        )


    def forward(self, phi1, phi2):
        f = self.focus(phi1)
        l = self.loss(phi2)

        return f, l


class MultiHeadAttention(Module):
    def __init__(self, in_ch, size):
        super(MultiHeadAttention, self).__init__()
        h   = params.model.att_trd.h
        dk  = params.model.att_trd.dk
        sim = params.model.att_trd.n_sim

        self.w_psi   = nn.Conv2d(in_ch, dk*h,  kernel_size=1, bias=False)
        self.w_theta = nn.Conv2d(in_ch, dk*h,  kernel_size=1, bias=False)
        self.w_v     = nn.Conv2d(dk*h,  in_ch, kernel_size=1, bias=False)
    
        self.norm = nn.BatchNorm1d(sim*size*size) if params.model.att_trd.attnorm else \
                    lambda x: x

        self.att_func = nn.Softmax(dim=2) if params.model.att_trd.att_func == 'softmax' else \
                        nn.Sigmoid()      if params.model.att_trd.att_func == 'sigmoid' else None

    def forward(self, x, phi):
        batch, _, hei, wid = map(int, x.shape)
        h       = params.model.att_trd.h
        dk      = params.model.att_trd.dk
        sim     = params.model.att_trd.n_sim
        batch //= sim

        psi = self.w_psi(x)                                     # (batch*sim, h*dk, hei, wid)
        psi = psi.reshape(batch, sim, h, dk, hei*wid)           # (batch, sim, h, dk, hei*wid)
        psi = psi.transpose(3, 4)                               # (batch, sim, h, hei*wid, dk)
        psi = psi.transpose(1, 2)                               # (batch, h, sim, hei*wid, dk)
        psi = psi.reshape(batch*h, sim*hei*wid, dk)             # (batch*h, sim*hei*wid, dk)

        theta = self.w_theta(phi)                               # (batch*sim, h*dk, hei, wid)
        theta = theta.reshape(batch, sim, h, dk, hei*wid)       # (batch, sim, h, dk, hei*wid)
        theta = theta.transpose(3, 4)                           # (batch, sim, h, hei*wid, dk)
        theta = theta.transpose(1, 2)                           # (batch, h, sim, hei*wid, dk)
        theta = theta.reshape(batch*h, sim*hei*wid, dk)         # (batch*h, sim*hei*wid, dk)

        theta = theta -  torch.mean(theta,    dim=2, keepdim=True)
        theta = theta / ((torch.sum(theta**2, dim=2, keepdim=True) + 1e-8)**0.5)

        r = torch.matmul(theta, theta.transpose(1, 2).contiguous())     # (batch*h, sim*hei*wid, sim*hei*wid)
        r = self.norm(r)                                        # (batch*h, sim*hei*wid, sim*hei*wid)
        r = self.att_func(r)                                    # (batch*h, sim*hei*wid, sim*hei*wid)
        u = torch.matmul(r, psi)                                # (batch*h, sim*hei*wid, dk)

        u = u.reshape(batch, h, sim, hei*wid, dk)               # (batch, h, sim, hei*wid, dk)
        u = u.transpose(3, 4)                                   # (batch, h, sim, dk, hei*wid)
        u = u.transpose(1, 2)                                   # (batch, sim, h, dk, hei*wid)
        u = u.reshape(batch*sim, h*dk, hei, wid)                # (batch*sim, h*dk, hei, wid)

        v = self.w_v(u)

        return v, r


class AttentionModule(Module):
    def __init__(self, in_ch, size):
        super(AttentionModule, self).__init__()

        self.rs = nn.Sequential(
            torchvision.models.resnet.BasicBlock(in_ch, in_ch),
            torchvision.models.resnet.BasicBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, in_ch*3, kernel_size=1),
            nn.ReLU(inplace=True),
            Lambda(lambda x: torch.chunk(x, 3, 1)),
        )

        self.focus    = FocusAttention(in_ch)
        self.relative = MultiHeadAttention(in_ch, size)

    def forward(self, x):
        phi1, phi2, phi3 = self.rs(x)

        f, l = self.focus(phi1, phi2)
        v, r = self.relative(x*f, phi3)

        y = x * f + v

        return y, l, f, r, phi2


def pretrained_resnet18():
    path  = params.model.trained_path.resnet18
    model = torchvision.models.resnet18()
    if params.model.use_trained:
        print('load trained', path)
        model.load_state_dict(torch.load(path, map_location='cpu'))
    return model


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()

        print('resnet18_trd_akn_ifs is used')

        n_class = params.dataset.n_class

        self.base    = pretrained_resnet18()
        self.at      = AttentionModule(256, 14)
        self.fc      = nn.Linear(512, n_class, bias=params.model.fc_bias)
        self.flatten = lambda x: torch.flatten(x, 1)

        self.lsmx = nn.LogSoftmax(dim=1)


    def forward(self, x, requires_attention=False):
        layers = [
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.maxpool,
            self.base.layer1,
            self.base.layer2,
            self.base.layer3,
            self.at,
            self.base.layer4,
            self.base.avgpool,
            self.flatten,
            self.fc,
            self.lsmx,
        ]

        batch, sim, ch, hei, wid = map(int, x.shape)
        x = x.reshape(batch*sim, ch, hei, wid)

        maps = []

        for l in layers:
            if l == self.at:
                x, h, f, r, phi = l(x)
            else:
                x = l(x)
            if l not in [
                self.base.avgpool,
                self.flatten,
                self.fc,
                self.lsmx,
            ]:
                maps.append(x)
        
        if requires_attention:
            return x, h, f, r, phi, maps

        return x, h



def create_train_engine(model:nn.Module, optimizer:optim.Optimizer, device:int=None):
    if device >= 0:
        model.to(device)
    
    loss_func = nn.CrossEntropyLoss()
    
    def _update(_, batch):
        model.train()

        x, t = batch
        if device >= 0:
            x = x.cuda(device, non_blocking=True)
            t = t.cuda(device, non_blocking=True)


        batch, sim, _, _, _ = map(int, x.shape)
        t = t.reshape(batch*sim)

        y, h = model(x)

        loss_pred = loss_func(y, t)
        loss_atts = loss_func(h, t)
        loss = loss_pred + loss_atts * params.model.att_trd.att_ratio

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        top1 = utils.TopN_accuracy(y, t, 1)
        top5 = utils.TopN_accuracy(y, t, 5)

        mets = {
            'pred':loss_pred.item(),
            'atts':loss_atts.item(),
            'top1':top1,
            'top5':top5
        }

        return mets
    
    eg = engine.Engine(_update)

    pred_avg = metrics.Average(lambda x: x['pred'])
    atts_avg = metrics.Average(lambda x: x['atts'])
    top1_avg = metrics.Average(lambda x: x['top1'])
    top5_avg = metrics.Average(lambda x: x['top5'])

    pred_avg.attach(eg, 'pred')
    atts_avg.attach(eg, 'atts')
    top1_avg.attach(eg, 'top1')
    top5_avg.attach(eg, 'top5')

    return eg


def create_eval_engine(model:nn.Module, device:int=None):
    if device >= 0:
        model.to(device)
    
    loss_func = nn.CrossEntropyLoss()
    
    def _update(_, batch):
        model.eval()

        x, t = batch
        if device >= 0:
            x = x.cuda(device, non_blocking=True)
            t = t.cuda(device, non_blocking=True)

        batch, sim, _, _, _ = map(int, x.shape)
        t = t.reshape(batch*sim)

        with torch.no_grad():
            y, h = model(x)

        loss_pred = loss_func(y, t)
        loss_atts = loss_func(h, t)

        top1 = utils.TopN_accuracy(y, t, 1)
        top5 = utils.TopN_accuracy(y, t, 5)

        mets = {
            'pred':loss_pred.item(),
            'atts':loss_atts.item(),
            'top1':top1,
            'top5':top5
        }
        
        return mets

    eg = engine.Engine(_update)

    pred_avg = metrics.Average(lambda x: x['pred'])
    atts_avg = metrics.Average(lambda x: x['atts'])
    top1_avg = metrics.Average(lambda x: x['top1'])
    top5_avg = metrics.Average(lambda x: x['top5'])

    pred_avg.attach(eg, 'pred')
    atts_avg.attach(eg, 'atts')
    top1_avg.attach(eg, 'top1')
    top5_avg.attach(eg, 'top5')

    return eg


def get_metrics_name():
    dst = ['pred', 'atts', 'top1', 'top5']
    return dst


def show_relative_attention():
    import os
    import sys
    import mylib.plot.colormap as cmap
    import mylib.utils.io as io

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    import datasets.dataset
    import datasets.load
    import datasets.preprocess


    model_path = sys.argv[1] if len(sys.argv) > 1 else input('model path... ')
    gpu        = sys.argv[2] if len(sys.argv) > 2 else input('gpu... ')
    gpu        = int(gpu)

    d, n = os.path.split(model_path)
    save_dir = os.path.join(d, '..', n[:-4])
    io.mkdir(save_dir)

    model = Model()
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    if gpu >= 0:
        model.to(gpu)
    model.eval()
    
    n_class = params.dataset.n_class
    sim     = params.model.att_trd.n_sim
    n       = 14
    head    = params.model.att_trd.h

    softmax = nn.Softmax(dim=1)
    log_smx = nn.LogSoftmax(dim=1)
    loss_func = nn.CrossEntropyLoss(reduction='none')


    def get_inputs():
        img_pathes = input('image path... ')
        if not img_pathes:
            sys.exit(0)

        if img_pathes[-4:] == '.txt':
            dirname    = os.path.dirname(img_pathes)
            img_pathes = io.load_text(img_pathes).strip().split('\n')
            index      = int(img_pathes[-1])
            img_pathes = [os.path.join(dirname, f) for f in img_pathes[:-1]]
        else:
            img_pathes = img_pathes.split(',')
            index = int(input('index... '))


        name_base  = os.path.join(save_dir, os.path.basename(img_pathes[0])[:-4])

        imgs = [datasets.load.load_image(img_path) for img_path in img_pathes]
        imgs = [datasets.preprocess.preprocess(img, False) for img in imgs]
        imgs = [imgs[i%len(imgs)] for i in range(sim)]

        x = torch.FloatTensor([imgs])
        if gpu >= 0:
            x = x.to(gpu)
        x.requires_grad_()
        
        return imgs, x, index, name_base, len(img_pathes)
    

    def get_predicted(x, imgs, length):
        y, h, f, r, phi, ms = model.forward(x, requires_attention=True)

        y = torch.exp(y)
        h = torch.exp(h)
        
        phi  = phi.reshape(sim, 256, n*n)
        phi  = phi.transpose(1, 2)
        phi  = phi.reshape(sim*n*n, 256)
        prb  = model.at.focus.loss[2](phi)
        lsmx = log_smx(prb)
        prb  = softmax(prb)

        y   = utils.as_numpy(y)
        h   = utils.as_numpy(h)
        prb = utils.as_numpy(prb)
        r   = utils.as_numpy(r)                      # (  h, sim*hei*wid, sim*hei*wid)
        f   = utils.as_numpy(f)                      # (sim, 1, hei, wid)

        imgs = [img.transpose(1, 2, 0) for img in imgs[:length]]
        prb  = prb.reshape(sim, n, n, n_class)
        r    = r.reshape(head, sim, n*n, sim,  n,  n)
        f    = f.reshape(sim, n, n)

        return y, h, prb, r, f, imgs, lsmx, ms
    

    def save_imgs(name_base, imgs):
        plt.figure(0, figsize=(16, 8))
        save_path = name_base + '_img.png'
        for i, img in enumerate(imgs):
            plt.subplot(2, 4, i+1)
            plt.imshow(img)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


    def save_prd_idx(name_base, imgs, prob, ys, hs):
        plt.figure(0, figsize=(16, 8))
        save_path = '{}_prob_idx.png'.format(name_base)
        for i, (img, p, y, h) in enumerate(zip(imgs, prob, ys, hs)):
            p = p[:,:,index]
            c = np.stack(cmap.jet(p, vmin=0, vmax=1), axis=2)
            c = np.uint8(c * 255)
            c = np.asarray(Image.fromarray(c).resize((224,224)), dtype=np.float32) / 255

            g = img*0.5 + c*0.5

            plt.subplot(2, 4, i+1)
            plt.imshow(g)

            title = 'y:{}({:.3f}) {}({:.3f})\nh:{}({:.3f}) {}({:.3f})' \
                        .format(np.argmax(y), y[np.argmax(y)], index, y[index],
                                np.argmax(h), h[np.argmax(h)], index, h[index])
            plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    
    def save_prd_prb(name_base, imgs, prob, ys, hs):
        plt.figure(0, figsize=(16, 8))
        save_path = '{}_prob_prd.png'.format(name_base)
        for i, (img, p, y, h) in enumerate(zip(imgs, prob, ys, hs)):
            p = p[:,:,np.argmax(h)]
            c = np.stack(cmap.jet(p, vmin=0, vmax=1), axis=2)
            c = np.uint8(c * 255)
            c = np.asarray(Image.fromarray(c).resize((224,224)), dtype=np.float32) / 255

            g = img*0.5 + c*0.5

            plt.subplot(2, 4, i+1)
            plt.imshow(g)

            title = 'y:{}({:.3f}) {}({:.3f})\nh:{}({:.3f}) {}({:.3f})' \
                        .format(np.argmax(y), y[np.argmax(y)], index, y[index],
                                np.argmax(h), h[np.argmax(h)], index, h[index])
            plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    
    def save_focus(name_base, imgs, focus):
        plt.figure(1, figsize=(16, 8))
        save_path = name_base + '_focus.png'
        for i, (img, a) in enumerate(zip(imgs, focus)):
            c = np.stack(cmap.jet(a, vmin=0, vmax=1), axis=2)
            c = np.uint8(c * 255)
            c = np.asarray(Image.fromarray(c).resize((224,224)), dtype=np.float32) / 255

            g = img*0.5 + c*0.5

            plt.subplot(2, 4, i+1)
            plt.imshow(g)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    

    def save_relate(name_base, imgs, relate, p, x, y):
        for h in range(head):
            plt.figure(figsize=(12, 6))
            plt.clf()
            save_path = name_base + '_relate{}_{}.png'.format(p, h)
            att = relate[h][0][p]
            att = att / np.max(att)
            for i, (img, a) in enumerate(zip(imgs, att)):
                c = np.stack(cmap.jet(a, vmin=0, vmax=1), axis=2)
                c = np.uint8(c * 255)
                c = np.asarray(Image.fromarray(c).resize((224,224)), dtype=np.float32) / 255

                g = img*0.5 + c*0.5

                plt.subplot(2, 4, i+1)
                plt.imshow(g)
                if i == 0:
                    for xs, ys, xe, ye in [[0,0,1,0], [0,0,0,1], [0,1,1,1], [1,0,1,1]]:
                        x0 = (x+xs) / n * 224
                        y0 = (y+ys) / n * 224
                        x1 = (x+xe) / n * 224
                        y1 = (y+ye) / n * 224
                        plt.plot([x0, x1], [y0, y1], color='white', linewidth=4)
                plt.title('max attention coefficient = {:.3f}'.format(np.max(a)))
            plt.tight_layout()
            plt.savefig(save_path)
        plt.show()


    def save_grad(name_base, imgs, inp, lsmx, p, x, y):
        plt.figure(3, figsize=(16, 8))
        plt.clf()
        save_path = '{}_grad_{}.png'.format(name_base, p)


        model.zero_grad()
        if inp.grad is not None:
            inp.grad *= 0

        lsmx = lsmx.reshape(sim, n, n, n_class)[:, y, x, :]
        prds = np.argmax(utils.as_numpy(lsmx), axis=1)
        prds = torch.LongTensor(prds).cuda(gpu)
        ls = torch.mean(loss_func(lsmx, prds))
        ls.backward(retain_graph=True)
        gs = inp.grad
        gs = utils.as_numpy(gs)[0]
        gs = np.sum(np.abs(gs), axis=1)
        for i, (img, g) in enumerate(zip(imgs, gs)):
            c = np.stack(cmap.jet(g), axis=2)
            c = np.uint8(c * 255)
            c = np.asarray(Image.fromarray(c).resize((224,224)), dtype=np.float32) / 255

            g = img*0.5 + c*0.5

            plt.subplot(2, 4, i+1)
            plt.imshow(g)
            for xs, ys, xe, ye in [[0,0,1,0], [0,0,0,1], [0,1,1,1], [1,0,1,1]]:
                x0 = (x+xs) / n * 224
                y0 = (y+ys) / n * 224
                x1 = (x+xe) / n * 224
                y1 = (y+ye) / n * 224
                plt.plot([x0, x1], [y0, y1], color='white', linewidth=4)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


    def save_middle(name_base, ms, length):
        for i in range(length):
            save_path = '{}_middle_{}.png'.format(name_base, i)
            plt.figure(figsize=(4.2*len(ms), 8))

            for j, m in enumerate(ms):
                m = m.detach().cpu().numpy()
                m_ = np.sum(m[i]**2, axis=0)**0.5
                plt.subplot(2, len(ms), j+1)
                plt.imshow(m_)
                plt.colorbar()

                plt.subplot(2, len(ms), j+1+len(ms))
                last = np.clip(m[i], 0, None)
                avg  = np.mean(last, axis=(1, 2), keepdims=True)
                avg  = avg - np.mean(avg)
                avg  = avg / np.sum(avg**2)**0.5
                last = last - np.mean(last, axis=0, keepdims=True)
                last = last / (np.sum(last**2, axis=0, keepdims=True)**0.5 + 1e-8)
                coef = np.sum(avg * last, axis=0)
                plt.imshow(coef, cmap='jet')
                plt.colorbar()

            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()


    while True:
        imgs, inp, index, name_base, length = get_inputs()
        
        ys, hs, prbs, relate, focus, imgs, lsmx, ms = get_predicted(inp, imgs, length)

        save_imgs(name_base, imgs)

        save_prd_prb(name_base, imgs, prbs, ys, hs)
        save_prd_idx(name_base, imgs, prbs, ys, hs)

        save_focus(name_base, imgs, focus)
        save_middle(name_base, ms, length)


        def onclick(e):
            if e.xdata is None or e.ydata is None:
                return
            x = int(e.xdata / 224 * n)
            y = int(e.ydata / 224 * n)
            p = x + y*n
            save_grad(name_base, imgs, inp, lsmx, p, x, y)
            save_relate(name_base, imgs, relate, p, x, y)


        fig = plt.figure(0)
        plt.imshow(imgs[0])
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()



if __name__ == "__main__":
    show_relative_attention()
