import os
import random

import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from . import preprocess


class Dataset(Dataset):
    def __init__(self, pathes, train, crop_size, dir_img, **_):
        self.pathes   = pathes
        self.train    = train
        self.img_size = crop_size
        self.dir_img  = dir_img

        dis = {}
        for f in pathes:
            index = f['LABEL']
            if index not in dis:
                dis[index] = []
            dis[index].append(f)
        self.disease = dis



    def triplet_sample(self, f):
        i = f['LABEL']
        p = random.choice(self.disease[i])
        return p


    def __len__(self) -> int:
        return len(self.pathes)


    def __getitem__(self, i:int):
        item1  = self.pathes[i]
        item2  = self.triplet_sample(item1)
        path1  = os.path.join(self.dir_img, item1['UJPG'])
        path2  = os.path.join(self.dir_img, item2['UJPG'])
        img1   = preprocess.preprocess(path1, self.train, self.img_size)
        img2   = preprocess.preprocess(path2, self.train, self.img_size)
        label1 = item1['LABEL']
        label2 = item2['LABEL']

        assert label1 == label2
        
        return img1, img2, label1
