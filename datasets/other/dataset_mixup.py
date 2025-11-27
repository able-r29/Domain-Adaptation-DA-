# T:\masaya\nsdd\PTLR\datasets\dataset_mixup.py
# からコピー

import random

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import ImageFile
from torch.utils.data import Dataset, DataLoader
import os
from . import preprocess

import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True



def get_pathes(n_fold, i_fold, path_src):
    print(i_fold, 'of', path_src, 'is used')

    folds   = utils.load_json(path_src)
    indices = np.arange(n_fold*2, dtype=int)
    indices = np.roll  (indices, i_fold*2)
    train   = indices[:(n_fold-1)*2]
    valid   = indices[-2:-1]
    test    = indices[-1:]

    train = [f for i in train for f in folds[i]]
    valid = [f for i in valid for f in folds[i]]
    test  = [f for i in test  for f in folds[i]]
    return train, valid, test



def get_dataset(n_fold, i_fold, path_src, device, batch_size, **karg_ds):
    train, val1, val2 = get_pathes(n_fold, i_fold, path_src)

    ds_train = ImageDataset(train, True,  device, **karg_ds)
    ds_valt  = ImageDataset(train, False, device, **karg_ds)
    ds_val1  = ImageDataset(val1,  False, device, **karg_ds)
    ds_val2  = ImageDataset(val2,  False, device, **karg_ds)

    karg_loader = dict(batch_size =batch_size,
                       shuffle    =True,
                       num_workers=4,
                       pin_memory=True)

    loader_train = DataLoader(ds_train, **karg_loader)
    loader_valt  = DataLoader(ds_valt,  **karg_loader)
    loader_val1  = DataLoader(ds_val1,  **karg_loader)
    loader_val2  = DataLoader(ds_val2,  **karg_loader)

    return loader_train, loader_valt, loader_val1, loader_val2



def get_transform(train, device, **karg):
    if not train:
        dst = nn.Sequential(
            T.ConvertImageDtype   (torch.float32),
            T.CenterCrop(karg['size_crop']),
            T.Resize    (karg['size_train'], T.InterpolationMode.NEAREST),
        ).to(device)
        return dst

    lays = [
            T.ConvertImageDtype   (torch.float32),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip  (0.5),
    ]
    if karg['gaussian'] is not None:
        lays.append(T.GaussianBlur(**karg['gaussian']))
    if karg['colorjitter'] is not None:
        lays.append(T.ColorJitter(**karg['colorjitter']))
    if karg['affine'] is not None:
        lays.append(T.RandomAffine(**karg['affine'],
                                   interpolation=T.InterpolationMode.BILINEAR))
    if karg['resizecrop'] is not None:
        lays.append(T.RandomResizedCrop(**karg['resizecrop'],
                                        interpolation=T.InterpolationMode.BILINEAR))
    else:
        lays.append(T.RandomCrop(karg['size_crop']))
    lays.append(T.Resize(karg['size_train'],
                         interpolation=T.InterpolationMode.NEAREST))

    dst = nn.Sequential(*lays).to(device)
    return dst


# class ImageDataset(Dataset):
class Dataset(Dataset):
    # def __init__(self, pathes, train, device, image_size, n_class, alpha, **karg):
    def __init__(self, pathes, train, image_size, dir_img, n_class, alpha, **karg):
        self.pathes   = pathes
        self.train    = train
        # self.device   = device
        # self.trans    = get_transform(train, device, **karg)
        self.img_size = image_size
        self.n_class  = n_class
        self.alpha    = alpha
        self.dir_img  = dir_img


    def __len__(self) -> int:
        return len(self.pathes)


    def get_metas(self, indices):
        indices = utils.as_numpy(indices)
        dst = [self.pathes[i] for i in indices]
        return dst


    def process(self, path):
        img = torchvision.io.read_image(path)
        assert min(img.shape[1:]) == self.img_size

        img = img.to(self.device)
        img = self.trans(img)
        img = img.cpu()

        return img


    def __getitem__(self, i:int):
        item1  = self.pathes[i]
        path1  = os.path.join(self.dir_img, item1['UJPG'])
        label1 = item1['LABEL']
        img1   = preprocess.preprocess(path1, self.train, self.img_size)

        item2  = random.choice(self.pathes)
        path2  = os.path.join(self.dir_img, item2['UJPG'])
        label2 = item2['LABEL']
        img2   = preprocess.preprocess(path2, self.train, self.img_size)

        # path1, label1 = self.pathes[i]
        # path2, label2 = random.choice(self.pathes)

        # img1 = self.process(path1)
        # img2 = self.process(path2)

        ratio = np.random.beta(self.alpha, self.alpha)

        img = ratio * img1 + (1-ratio) * img2

        label = np.zeros(self.n_class)
        label[label1] = ratio
        label[label2] = 1 - ratio

        return img, label
