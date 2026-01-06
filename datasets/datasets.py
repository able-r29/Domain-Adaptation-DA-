import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import utils
# from . import dataset_simple, dataset_meta, dataset_mtnv, dataset_mtel, dataset_triplet, dataset_meta_triplet, dataset_mixup, dataset_meta_triplet_mixup, dataset_quality, dataset_meta_triplet_quality, dataset_meta_triplet_quality_disease, dataset_meta_quality_disease, ViT, dataset_meta_triplet_case_num
from . import dataset_simple, dataset_meta, dataset_triplet, dataset_meta_triplet


module_map = {
                # 'mtnv'     : dataset_mtnv,
                # 'mtel'     : dataset_mtel,
                'meta'     : dataset_meta,
                'simple'   : dataset_simple,
                'triplet'  : dataset_triplet,
                'meta_trp' : dataset_meta_triplet,
                # 'mixup'    : dataset_mixup,
                # 'meta_trip_mixup' : dataset_meta_triplet_mixup,
                # 'quality' : dataset_quality,
                # 'meta_trip_qual' : dataset_meta_triplet_quality,
                # 'meta_trip_qual_dis' : dataset_meta_triplet_quality_disease,
                # 'meta_qual_dis' : dataset_meta_quality_disease,
                # 'ViT' : ViT,
                # 'meta_trip_case_num' : dataset_meta_triplet_case_num
             }


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)



def get_dataset_single(name, pathes, train, batch_size, n_workers, shuffle, generator=None, sampler=None, **karg_ds):
    module = module_map[name]

    ds = module.Dataset(pathes=pathes, train=train,  **karg_ds)

    init_fn = None if generator is None else seed_worker

    if train:
        if hasattr(ds, 'sampler'):
            sampler = ds.sampler() # ミニバッチのサンプリング方法を変更
            shuffle = False # WeightedRandomSamplerを使う場合はそっちでシャッフルしてくれるので，shuffle = Falseとしている


    loader = DataLoader(ds,
                        batch_size    =batch_size,
                        num_workers   =n_workers,
                        pin_memory    =True,
                        generator     =generator,
                        shuffle       =shuffle,
                        sampler       = sampler,
                        worker_init_fn=init_fn)

    return loader


def get_dataset(name, i_fold, path_src, batch_size, n_workers, shuffle, generator=None, **karg_ds):
    """
    データセット取得（修正版）
    
    戻り値:
    - loader_train: 学習用データ
    - loader_val: 検証用データ（評価用）
    """
    # ★ ホールドアウトの場合
    if i_fold is None:
        if isinstance(path_src, dict):
            # 辞書形式（train/validationが分かれている）
            train_data = utils.load_json(path_src['train'])
            val_data = utils.load_json(path_src['validation'])
            
            karg_loader = dict(name=name, batch_size=batch_size, n_workers=n_workers, 
                             generator=generator, shuffle=shuffle, **karg_ds)
            
            loader_train = get_dataset_single(pathes=train_data, train=True, **karg_loader)
            loader_val = get_dataset_single(pathes=val_data, train=False, **karg_loader)
            
            # ★ 修正: 2つの値のみ返す
            return loader_train, loader_val
        else:
            # 従来の単一ファイル（推論用など）
            folds = utils.load_json(path_src)
            loader = get_dataset_single(name=name, pathes=folds, train=False, 
                                      batch_size=batch_size, n_workers=n_workers, 
                                      generator=generator, shuffle=shuffle, **karg_ds)
            return loader, None  # ★ 2つ返すように統一
    
    # ★ K-fold cross validationの場合
    folds = utils.load_json(path_src)
    assert i_fold < len(folds)

    indices = np.arange(len(folds), dtype=int)
    indices = np.roll(indices, i_fold)
    train   = indices[:len(folds)-1]
    val     = indices[-1:]

    train = [f for i in train for f in folds[i]]
    val   = [f for i in val   for f in folds[i]]

    karg_loader = dict(name      =name,
                       batch_size=batch_size,
                       n_workers =n_workers,
                       generator =generator,
                       i_fold    =i_fold,
                       shuffle   =shuffle,
                       **karg_ds)

    loader_train = get_dataset_single(pathes=train, train=True,  **karg_loader)
    loader_val = get_dataset_single(pathes=val,   train=False, **karg_loader)

    # ★ 修正: 2つの値のみ返す
    return loader_train, loader_val
