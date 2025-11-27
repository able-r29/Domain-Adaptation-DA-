import os

import numpy as np
from torch.utils.data import Dataset

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



def meta_naive_bayes(items, use):
    vc = {}
    for item in items:
        m = meta_preprocess(item, use)
        m = ''.join(map(str, m.astype(np.int)))
        if m not in vc:
            vc[m] = np.zeros(59, dtype=np.float32)
        vc[m][item['LABEL']] += 1
    vc = {k:v/np.sum(v) for k, v in vc.items()}
    return vc



class Dataset(Dataset):
    def __init__(self, pathes, train, crop_size, dir_img, meta_use, **_):
        self.pathes   = pathes
        self.train    = train
        self.img_size = crop_size
        self.dir_img  = dir_img
        self.meta_use = meta_use
        self.probs    = meta_naive_bayes(pathes, meta_use)


    def __len__(self) -> int:
        return len(self.pathes)


    def __getitem__(self, i:int):
        item  = self.pathes[i]
        path  = os.path.join(self.dir_img, item['UJPG'])
        label = item['LABEL']
        img   = preprocess.preprocess(path, self.train, self.img_size)

        if not self.meta_use:
            return img, label

        meta = meta_preprocess(item, self.meta_use)

        key = ''.join(map(str, meta.astype(np.int)))
        prb = self.probs[key]

        return img, label, meta, prb


if __name__ =='__main__':
    import matplotlib.pyplot as plt
    import mylib.utils.io as io
    folds = io.load_json(io.load_args('train fold:')[0])
    for fold in folds:
        prob = meta_naive_bayes(fold, ['age', 'sex', 'part'])
        for k, v in prob.items():
            plt.bar(range(len(v)), v)
            plt.title(k)
            plt.show()
