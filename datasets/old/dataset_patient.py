import random

import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import params
import utils
from . import load, preprocess


def format_params() -> str:
    fmt = 'dataset_rmder_{}_{}_blar{}_img{}-{}'

    target    = params.dataset.target
    name      = params.dataset.name
    blar      = params.dataset.augmentation.gaussian
    img_size  = params.dataset.size
    crop_size = params.model.image_size
    
    dst = fmt.format(target, name, blar, img_size, crop_size)
    return dst


class PatientDataset(Dataset):
    def __init__(self, files, train, multishot=False, need_index=False, set_shuffle=False, imgs=None):
        for i, f in enumerate(files):
            f['index'] = i

        dis = {}
        for f in files:
            name = f['disease']
            uid  = f['id']
            if name not in dis:
                dis[name] = {}
            if uid not in dis[name]:
                dis[name][uid] = []
            dis[name][uid].append(f)
        dis = [[dis[n][u] for u in dis[n]] for n in dis]
        self.disease = dis

        pat = {}
        for f in files:
            name = f['disease']
            uid  = f['id']
            if uid not in pat:
                pat[uid] = {}
            if name not in pat[uid]:
                pat[uid][name] = []
            pat[uid][name].append(f)
        pat = [pat[u][n] for u in pat for n in pat[u]]
        self.patients = pat

        n_sim = params.model.att_trd.n_sim
        sets = []
        for p in pat:
            n_img = len(p)
            rand_index = list(range(n_img))
            if set_shuffle:
                temp = []
                s = n_img // n_sim + 1
                for i in range(s):
                    temp.extend(rand_index[i::s])
                assert set(rand_index) == set(temp)
                rand_index = temp
            for top in range(0, n_img, n_sim):
                img_set = []
                for index in range(top, top+n_sim):
                    index = index % n_img
                    index = rand_index[index]
                    img_set.append(p[index])
                sets.append(img_set)
        self.img_sets = sets


        self.fold_files = files
        self.train      = train
        self.multishot  = multishot
        self.need_index = need_index
        self.name2index = load.load_name2index()

        print('dataset:', len(files), 'files')

        self.imgs = load.load_pickled() if imgs is None else imgs

        self.sample_func = {
                                'oversample':self.oversample,
                                'normal'    :self.normal_sample
                            }[params.dataset.sample if train else 'normal']
        print('dataset:', len(files), 'files')
        print('        ', self.sample_func,__name__)


    def oversample(self, _):
        d = random.choice(self.disease)
        p = random.choice(d)
        dst = random.choices(p, k=params.model.att_trd.n_sim)
        return dst


    def normal_sample(self, i):
        if self.train:
            p   = random.choice(self.patients)
            dst = random.choices(p, k=params.model.att_trd.n_sim)
        else:
            dst = self.img_sets[i]
        return dst


    def __len__(self) -> int:
        return len(self.img_sets)


    def get_metas(self, indices):
        indices = utils.as_numpy(indices)
        dst = [self.fold_files[i] for i in indices]
        return dst


    def __getitem__(self, i:int):
        files   = self.sample_func(i)
        images  = []
        truths  = []
        indices = []

        for f in files:
            path    = f['path']
            if params.dataset.path_replace is not None:
                path = path.replace(*params.dataset.path_replace)
            disease = f['disease']
            index   = f['index']

            src = load.get_cached_path(path) if self.imgs is None else self.imgs[path]
            img = Image.open(src)

            if self.multishot:
                img = preprocess.multishot_preprocess(img)
            else:
                img = preprocess.preprocess(img, self.train)
            images.append(img)

            truth = self.name2index[disease]
            truths.append(truth)

            indices.append(index)
        
        images  = np.array(images,  dtype=np.float32)
        truths  = np.array(truths,  dtype=np.int64)
        indices = np.array(indices, dtype=np.int64)


        if self.need_index:
            return images, truths, indices
        
        return images, truths


# if __name__ == "__main__":
#     ds = PatientDataset(load.load_fold(0)[0], False)
#     for s in ds.img_sets:
#         print(len(s))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    path = input('path... ')

    img = load.load_image(path)
    while True:
        pps = preprocess.preprocess(img)
        pps = np.transpose(pps, (1, 2, 0))
        vvs = preprocess.preprocess(img, False)
        vvs = np.transpose(vvs, (1, 2, 0))
        plt.figure(0, figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(pps)
        plt.subplot(1, 2, 2)
        plt.imshow(vvs)
        plt.show()
    


if __name__ == "__main__":
    import io
    import tqdm

    path = input('path... ')

    img = Image.open(path)
    img = img.convert('RGB')
    img = img.resize((410, 256), resample=Image.BILINEAR)
    bio = io.BytesIO()
    img.save(bio, format='JPEG')

    for i in tqdm.trange(1000):
        img = Image.open(bio)
        img = np.asarray(img, dtype=np.float32)
        preprocess(img)
    