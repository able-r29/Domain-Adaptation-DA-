import os
import random

import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from . import preprocess


def meta_preprocess(metas, use):
    #11
    def age():
        dst = np.zeros(11, dtype=np.float32)
        if len(metas['age']) != 1:
            val = 10
        else:
            val = metas['age'][0]
            val = min(int(val)//10, 9) if val.isdigit() and int(val) < 120 else 10
        dst[val] = 1
        return dst

    #3
    def sex():
        if len(metas['sex']) != 1:
            val = 2
        elif metas['sex'][0] == 'male':
            val = 0
        elif metas['sex'][0] == 'female':
            val = 1
        else:
            val = 2
        dst = np.zeros(3, dtype=np.float32)
        dst[val] = 1
        return dst

    #7
    def part():
        table = {
            'unknown'   :0,
            'whole body':1,
            'scalp'     :2,
            'face'      :3,
            'upper arm' :4,
            'upperarm'  :4,
            'trunk'     :5,
            'leg'       :6
        }
        dst = np.zeros(7, dtype=np.float32)
        for v in metas['part']:
            v = v.lower()
            if v in table:
                v = table[v]
            else:
                v = 0
            dst[v] = 1
        return dst

    funcs = {
        'age' :age,
        'sex' :sex,
        'part':part
    }
    dst = [funcs[u]() for u in use]
    return np.concatenate(dst)



class Dataset(Dataset):
    def __init__(self, pathes, train, crop_size, dir_img, meta_use, **_):
        self.pathes   = pathes
        self.train    = train
        self.img_size = crop_size
        self.dir_img  = dir_img
        self.meta_use = meta_use

        dis = {}
        for f in pathes:
            index = f['QUAL_BIN']
            if index not in dis:
                dis[index] = []
            dis[index].append(f)
        self.disease = dis



    def triplet_sample(self, f):
        i = f['QUAL_BIN']
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
        label1 = item1['QUAL_BIN']
        label2 = item2['QUAL_BIN']
        meta1  = meta_preprocess(item1, self.meta_use)
        meta2  = meta_preprocess(item2, self.meta_use)

        assert label1 == label2


        return img1, img2, meta1, meta2, label1
