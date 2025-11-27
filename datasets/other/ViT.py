import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from . import preprocess

def preprocess_ViT(path, train, size, crop_size):
    img = Image.open(path)

    width, height = img.size
    if width > height:
        width  = int(width / height * size)
        height = size
    else:
        height = int(height / width * size)
        width  = size

    img = img.resize((width, height), Image.BICUBIC)
    img = np.array(img).astype(np.float32)

    h, w, _ = img.shape
    if h != crop_size or w != crop_size:
        x = int((np.random.random() if train else 0.5) * (w-crop_size))
        y = int((np.random.random() if train else 0.5) * (h-crop_size))
        img = img[y:y+crop_size, x:x+crop_size, :]

    if train:
        if np.random.random() < 0.5:
            img = img[:,::-1,:]
        if np.random.random() < 0.5:
            img = img[::-1,:,:]
        if np.random.random() < 0.5:
            img = np.transpose(img, axes=(1, 0, 2))

    img /= 255
    img  = img.transpose((2, 0, 1))
    img  = img.copy()
    return img


class Dataset(Dataset):
    def __init__(self, pathes, train, image_size, crop_size, dir_img, **_):
        self.pathes   = pathes
        self.train    = train
        self.img_size = image_size
        self.crop_size = crop_size
        self.dir_img  = dir_img


    def __len__(self) -> int:
        return len(self.pathes)


    def __getitem__(self, i:int):
        item  = self.pathes[i]
        path  = os.path.join(self.dir_img, item['UJPG'])
        label = item['LABEL']
        img   = preprocess_ViT(path, self.train, self.img_size, self.crop_size)
        return img, label
