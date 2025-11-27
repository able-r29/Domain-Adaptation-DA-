import os

from torch.utils.data import Dataset

from . import preprocess



class Dataset(Dataset):
    def __init__(self, pathes, train, image_size, dir_img, **_):
        self.pathes   = pathes
        self.train    = train
        self.img_size = image_size
        self.dir_img  = dir_img


    def __len__(self) -> int:
        return len(self.pathes)


    def __getitem__(self, i:int):
        item  = self.pathes[i]
        path  = os.path.join(self.dir_img, item['UJPG'])
        label = item['QUAL_BIN']
        img   = preprocess.preprocess(path, self.train, self.img_size)

        return img, label
