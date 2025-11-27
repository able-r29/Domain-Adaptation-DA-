import json
import os
import pickle
import io
import time
from typing import Dict, Tuple, List

import numpy as np
import tqdm
from joblib import Parallel, delayed
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import params
import utils


def load_image(src:str, size=None) -> ImageFile.ImageFile:
    if size is None:
        size = params.dataset.size
    img  = Image.open(src)
    width, height = img.size
    if min(width, height) != size:
        if width > height:
            h = size
            w = width / height * h
        else:
            w = size
            h = height / width * w
        img = img.resize((int(w), int(h)), resample=Image.BILINEAR)
    return img


def get_cached_path(src_path:str, img_size:int=None):
    if img_size is None:
        img_size = params.dataset.size
    temp_dir = '/tmp/img{}'.format(img_size)
    src_path = src_path.replace('/home/users/Share/image/dermatology/20190917', '/home/users/Share/image/dermatology/NSDD-20190917')
    dst_path = src_path.replace('/home/users/Share/image/dermatology/NSDD-20190917', temp_dir)
    if os.path.exists(dst_path):
        return dst_path

    dir_names = []
    dir_name  = os.path.dirname(dst_path)
    while dir_name.count('/') > 1:
        if os.path.exists(dir_name):
            break
        dir_names.append(dir_name)
        dir_name = os.path.dirname(dir_name)
    
    for d in dir_names[::-1]:
        if not os.path.exists(d):
            os.mkdir(d)
    

    img = load_image(src_path, img_size)
    dst_file = open(dst_path, 'wb')
    img.save(dst_file, 'JPEG')
    dst_file.flush()
    dst_file.close()
    time.sleep(0.5)

    return dst_path


def get_tmpfs_path(src_path:str):
    temp_dir = params.dataset.tempfs_path
    fname    = os.path.basename(src_path)
    dst_path = os.path.join(temp_dir, fname)
    return dst_path


def load_pickled() -> Dict[str, io.BytesIO]:
    path = params.dataset.pickled_path
    if path is None:
        print('pickled object is not used')
        return None
    if not os.path.exists(path):
        path = params.dataset.pickled_pathes['local'][params.dataset.size]

    print('start: loading', path)

    with open(path, 'rb') as f:
        img_buf = pickle.load(f)

    print('pickled', len(img_buf), 'files are loaded from', path)

    return img_buf


def load_fold(i_fold:int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    assert i_fold < params.dataset.n_fold

    print(i_fold, 'of', params.dataset.fold_file, 'is used')
    with open(params.dataset.fold_file, 'r', encoding='utf_8_sig') as f:
        files = json.load(f)

    indices = np.arange(params.dataset.n_fold*2, dtype=int)
    indices = np.roll(indices, i_fold*2)
    train   = indices[:(params.dataset.n_fold-1)*2]
    valid   = indices[-2:-1]
    test    = indices[-1:]

    train = [f for i in train for f in files[i]]
    valid = [f for i in valid for f in files[i]]
    test  = [f for i in test  for f in files[i]]
    return train, valid, test


def load_name2index() -> Dict[str, int]:
    with open(params.dataset.name2index, 'r', encoding='utf-8_sig') as f:
        cvt = json.load(f)
    return cvt


def load_tree_conv() -> Dict[str, int]:
    with open(params.dataset.tree_file, 'r', encoding='utf-8_sig') as f:
        cvt = json.load(f)
    cvt = cvt[params.model.regulation.level]
    return cvt


if __name__ == "__main__":
    imgsize    = int(input('img size: '))
    files_path = 'metas/dataset_metas/files.json'
    files = utils.load_json(files_path)

    pathes = []
    for dis in files.values():
        for pat in dis.values():
            for img in pat:
                path = img['path']
                pathes.append(path)
    
    [get_cached_path(path, imgsize) for path in tqdm.tqdm(pathes)]
