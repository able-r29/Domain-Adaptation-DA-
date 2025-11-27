import os

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from . import preprocess
import numpy as np



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
        label = item['LABEL']
        img   = preprocess.preprocess(path, self.train, self.img_size)

        return img, label

    def batch_sampler():
        batch_sampler = BatchSampler_case
        return batch_sampler





class BatchSampler_case(BatchSampler):
    def __init__(self, meta):
        case_ujpg = {}
        for m in meta:
            ujpg = m['UJPG']
            case = m['CASE']
            label = m['LABEL']
            if case not in case_ujpg:
                case_ujpg = []
            case_ujpg[case].append(ujpg)

        ujpg_list = []
        for case, ujpg in case_ujpg.items():
            ujpg_list.append([ujpg, label])

        matrix2d = np.array([[1,2,3],
                            [4,5,6],
                            [7,8,9]])
        rng = np.random.default_rng()
        matrix2d = rng.permuted(matrix2d, axis=1) # 二次元目（列）を行ごとにシャッフル
        print(matrix2d)


        # loader = DataLoader(dataset)
        # self.labels_list = []
        # for _, label in loader:
        #     self.labels_list.append(label)
        # self.labels = torch.LongTensor(self.labels_list)
        # self.labels_set = list(set(self.labels.numpy()))
        # self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
        #                          for label in self.labels_set}
        # for l in self.labels_set:
        #     np.random.shuffle(self.label_to_indices[l])
        # self.used_label_indices_count = {label: 0 for label in self.labels_set}
        # self.count = 0
        # self.n_classes = n_classes
        # self.n_samples = n_samples
        # self.dataset = dataset
        # self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size