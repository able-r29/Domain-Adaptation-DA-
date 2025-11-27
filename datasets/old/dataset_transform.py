import random
import shutil
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import Normalize

import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True



def get_pathes(n_fold, i_fold, path_src):
    print(i_fold, 'of', path_src, 'is used')
    shutil.copy2(path_src, 'path_src.json')

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
                       num_workers=4,
                       pin_memory=True)

    loader_train = DataLoader(ds_train, **karg_loader, shuffle=True)
    loader_valt  = DataLoader(ds_valt,  **karg_loader, shuffle=False)
    loader_val1  = DataLoader(ds_val1,  **karg_loader, shuffle=False)
    loader_val2  = DataLoader(ds_val2,  **karg_loader, shuffle=False)

    return loader_train, loader_valt, loader_val1, loader_val2



class NormalizedSample:
    def __init__(self, src) -> None:
        print('normalized sample was used')
        self.length = len(src)
        self.items  = {}
        for p, l in src:
            if l not in self.items:
                self.items[l] = []
            self.items[l].append((p, l))

    def __getitem__(self, _) -> Tuple[str, int]:
        dst = random.choice(random.choice(self.items))
        return dst
    
    def __len__(self):
        return self.length



def get_transform(train, device, **karg):
    if not train:
        dst = nn.Sequential(
            T.ConvertImageDtype   (torch.float32),
            T.Normalize           (0, 255),
            T.CenterCrop(karg['size_crop']),
            T.Resize    (karg['size_train'], T.InterpolationMode.NEAREST),
        ).to(device)
        return dst

    lays = [
            T.ConvertImageDtype   (torch.float32),
            T.Normalize           (0, 255),
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


class ImageDataset(Dataset):
    def __init__(self, files, train, device, image_size, n_class, sample='normal', **karg):
        self.pathes   = NormalizedSample(files) \
                        if train and sample == 'normalized' else \
                        files
        self.train    = train
        self.device   = device
        self.trans    = get_transform(train, device, **karg)
        self.img_size = image_size
        self.n_class  = n_class
        self.sample   = sample


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
        path = self.pathes[i]
        if isinstance(path, str):
            path, label = path, None
        else:
            path, label = path
        img = self.process(path)

        if label is None:
            return img

        return img, label
