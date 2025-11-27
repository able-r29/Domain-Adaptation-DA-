import os
import random

import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from . import preprocess


def meta_preprocess(metas, use, case_num_):
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

    #1
    def case_num():
        dst = np.zeros(19, dtype=np.float32)
        if case_num_ < 10: # 症例の画像枚数が1~9枚 → 1区切りで1~9のどこかが1
            dst[case_num_-1] = 1
        else: # 症例の画像枚数が10枚以上 → 10区切りで10~20のどこかが1 100枚以上なら20が1
            num = min(int(case_num_)//10, 10)
            dst[num-1+9] = 1
        return dst

    funcs = {
        'age' :age,
        'sex' :sex,
        'part':part,
        'case_num':case_num
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
            index = f['LABEL']
            if index not in dis:
                dis[index] = []
            dis[index].append(f)
        self.disease = dis

        case_num = {}
        for f in pathes:
            case = f['CASE']
            if case not in case_num:
                case_num[case] = []
            case_num[case].append(f)

        ujpg_case_num = {}
        for f in pathes:
            ujpg = f['UJPG']
            case = f['CASE']
            ujpg_case_num[ujpg] = len(case_num[case])
        self.case_num = ujpg_case_num


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
        case_num1 = self.case_num[item1['UJPG']]
        case_num2 = self.case_num[item2['UJPG']]
        meta1  = meta_preprocess(item1, self.meta_use, case_num1)
        meta2  = meta_preprocess(item2, self.meta_use, case_num2)

        assert label1 == label2


        return img1, img2, meta1, meta2, label1
