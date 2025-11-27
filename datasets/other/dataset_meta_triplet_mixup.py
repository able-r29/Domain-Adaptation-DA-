import os
import random

import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from . import preprocess
# import preprocess
import torch.nn.functional as F
import torch

import collections
import numpy as np


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


def meta_class_str(f):
    label = f['LABEL']
    meta = meta_preprocess(f, ['age', 'sex', 'part']).astype(np.int64)
    meta = list(map(str, meta))
    meta = ''.join(meta)
    meta_label = f'{meta}_{label}'
    return meta_label


def mixup_img(img1, img2, label1, label2, ratio, n_class):
    # label1=F.one_hot(label1, n_class) #とりあえず同じクラスでmixupするので，labelはそのまま
    # label2=F.one_hot(label2, n_class)
    img = ratio * img1 + (1-ratio) * img2
    label = ratio * label1 + (1-ratio) * label2
    # return img, label

    assert label1 == label2
    return img, label1 #とりあえず同じクラスでmixupするので，labelはそのまま


def make_weights_for_balanced_classes(pathes, head, middle, tail): #ミニバッチを作るときに，middleとtailは多めにサンプリング 　各画像をサンプリングする重みの配列を作る
    weight = [0] * len(pathes)
    for idx, item in enumerate(pathes):
        label = item['LABEL']
        if label in head[0]:
            lambda_ = head[1]
        elif label in middle[0]:
            lambda_ = middle[1]
        elif label in tail[0]:
            lambda_ = tail[1]
        weight[idx] = lambda_
    return weight



def save_img(path, img):
    img  = img.transpose((1, 2, 0))
    img = np.uint8(img * 255)
    img = Image.fromarray(img)
    img.save(path)

class Dataset(Dataset):
    def __init__(self, pathes, train, crop_size, dir_img, meta_use, n_class, alpha, minibat_sample, mixup_ratio, **_):
        self.pathes   = pathes
        self.train    = train
        self.img_size = crop_size
        self.dir_img  = dir_img
        self.meta_use = meta_use
        self.n_class = n_class
        self.alpha = alpha
        self.mixup_ratio = mixup_ratio

        self.head = minibat_sample[0]
        self.middle = minibat_sample[1]
        self.tail= minibat_sample[2]

        dis = {}
        for f in pathes:
            index = f['LABEL']
            if index not in dis:
                dis[index] = []
            dis[index].append(f)
        self.disease = dis

        meta_class = {}
        for f in pathes:
            str = meta_class_str(f)
            if str not in meta_class:
                meta_class[str] = []
            meta_class[str].append(f)
        self.meta_class = meta_class


    def getitem_triplet(self, i:int):
        item1  = self.pathes[i]
        item2  = self.triplet_sample(item1)
        path1  = os.path.join(self.dir_img, item1['UJPG'])
        path2  = os.path.join(self.dir_img, item2['UJPG'])
        img1   = preprocess.preprocess(path1, self.train, self.img_size)
        img2   = preprocess.preprocess(path2, self.train, self.img_size)
        label1 = item1['LABEL']
        label2 = item2['LABEL']
        meta1  = meta_preprocess(item1, self.meta_use)
        meta2  = meta_preprocess(item2, self.meta_use)

        assert label1 == label2
        return img1, img2, meta1, meta2, label1

    def triplet_sample(self, f):
        i = f['LABEL']
        p = random.choice(self.disease[i])
        return p

    def getitem_mixup(self, item1):
        item2  = self.mixup_sample(item1)
        path1  = os.path.join(self.dir_img, item1['UJPG'])
        path2  = os.path.join(self.dir_img, item2['UJPG'])
        img1   = preprocess.preprocess(path1, self.train, self.img_size)
        img2   = preprocess.preprocess(path2, self.train, self.img_size)
        label1 = torch.tensor(item1['LABEL'])
        label2 = torch.tensor(item2['LABEL'])

        ratio = np.random.beta(self.alpha, self.alpha)
        img, label = mixup_img(img1, img2, label1, label2, ratio, self.n_class)
        return img, label, item1, item2

    def mixup_sample(self, f):
        meta_class = meta_class_str(f)
        # item2 = random.choice(self.pathes) #mixupのペアをランダムに選択
        item2 = random.choice(self.meta_class[meta_class]) #mixupのペアをクラスとメタデータが同じ画像から選択
        return item2

    # ミニバッチ作成時にクラスごとにサンプル数を変更し，モデルに見せる頻度を変える
    def sampler(self):
        weights = make_weights_for_balanced_classes(self.pathes, self.head, self.middle, self.tail)
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        return sampler


    def __len__(self) -> int:
        return len(self.pathes)
        # print(self.train)
        # print('len:100')
        # return 100


    def __getitem__(self, i:int):
        #trainの時のみmixupを行う
        # middle_classまたはtail_classの時のみmixupを行う
        if self.train:
            item1  = self.pathes[i]
            label1 = item1['LABEL']

            if label1 in self.middle[0] or label1 in self.tail[0]:
                if np.random.random() < self.mixup_ratio: #middle_classまたはtail_classの時,mixup_ratioの確率でmixupを行う
                    # print('mixupあり')
                    anchor  = self.pathes[i]
                    positive  = self.triplet_sample(anchor)

                    a_img_mix, a_label_mix, a_item1, a_item2 = self.getitem_mixup(anchor)
                    a_meta  = meta_preprocess(a_item1, self.meta_use)
                    a_meta_  = meta_preprocess(a_item2, self.meta_use)
                    assert np.array_equal(a_meta, a_meta_) #とりあえず同じメタデータでmixupする

                    p_img_mix, p_label_mix, p_item1, p_item2  = self.getitem_mixup(positive)
                    p_meta  = meta_preprocess(p_item1, self.meta_use)
                    p_meta_  = meta_preprocess(p_item2, self.meta_use)
                    assert np.array_equal(p_meta, p_meta_) #とりあえず同じメタデータでmixupする

                    assert torch.equal(a_label_mix, p_label_mix)
                    return a_img_mix, p_img_mix, a_meta, p_meta, int(a_label_mix)

        # print('mixupなし')
        item1  = self.pathes[i]
        item2  = self.triplet_sample(item1)
        path1  = os.path.join(self.dir_img, item1['UJPG'])
        path2  = os.path.join(self.dir_img, item2['UJPG'])
        img1   = preprocess.preprocess(path1, self.train, self.img_size)
        img2   = preprocess.preprocess(path2, self.train, self.img_size)
        label1 = item1['LABEL']
        label2 = item2['LABEL']
        meta1  = meta_preprocess(item1, self.meta_use)
        meta2  = meta_preprocess(item2, self.meta_use)

        assert label1 == label2
        return img1, img2, meta1, meta2, label1






if __name__ =='__main__':
    import mylib.utils.io as io
    dataset = io.load_json('t:/fujiwara/experiment/NSDD/dataset/fold/2021_11_20/train.json')
    if isinstance(dataset, list):
        if isinstance(dataset[0], list):
            dataset = sum(dataset, start=[])

    if isinstance(dataset, dict):
        dataset = dataset.values()

    dir_img = "D:/fujiwara/experiment/NSDD/dataset/2022_03_14/2022_03_14__512"
    meta_use = ["age", "sex", "part"]
    dataset = Dataset(pathes=dataset, train=True, crop_size=512, dir_img=dir_img, meta_use=meta_use, n_class=59, alpha=1)
    n_data = len(dataset)


    # while True:
    #     i = random.randint(0, len(dataset))
    #     a_img_mix, p_img_mix, a_meta, p_meta, a_label_mix = dataset[i]


    while True:
        i = random.randint(0, len(dataset))
        item = dataset.pathes[i]
        img, label, item1, item2 = dataset.getitem_mixup(item)
        print('meta')
        print(meta_preprocess(item1, meta_use))
        print(meta_preprocess(item2, meta_use))
        save_img("../debug/3_mix.png", img)
        print('mix label:', label)
        input()

    # alpha = 1
    # while True:
    #     path1 = '../debug/arrow_230_30_40.png'
    #     path2 = '../debug/arrow_40_20_200.png'
    #     img1   = preprocess.preprocess(path1, train=True, size=512)
    #     img2   = preprocess.preprocess(path2, train=True, size=512)
    #     # ratio = np.random.beta(alpha, alpha)
    #     ratio = float(input('ratio: '))
    #     print(ratio)
    #     label1 = torch.tensor(0)
    #     label2 = torch.tensor(1)
    #     img, label = mixup_img(img1, img2, label1, label2, ratio, 2)
    #     save_img(f"../debug/arrow_mix_ratio_{ratio}.png", img)

