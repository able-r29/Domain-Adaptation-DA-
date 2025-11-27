import argparse
import glob
import json
import os
import pickle
import sys

import matplotlib
import numpy as np
import torch
import tqdm
import pathlib

import datasets.datasets as dataset
import models.models as models
import utils

matplotlib.use('Agg')



def load_json(path):
    with open(path, 'r', encoding='utf_8') as f:
        dst = json.load(f)
    return dst



def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)



def parse_path(dir_result):
    if not os.path.exists(dir_result):
        return None

    path_config = os.path.join(dir_result, 'config.json')
    pathes_model = glob.glob(os.path.join(dir_result, '**.pt'))
    if len(pathes_model) != 1:
        print('length of saved model is not 1, but', len(pathes_model))
        return None

    path_result = os.path.join(os.path.dirname(dir_result), 'analyze', 'predict_val')

    return path_config, pathes_model[0], path_result



def predict(loader, model, device, preprocess):
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        dst_ys = None
        dst_t = []
        for batch in tqdm.tqdm(loader, total=len(loader), leave=False):
            x, t = preprocess(batch, device, True)
            ys = model.predict(x)
            if isinstance(ys, torch.Tensor):
                ys = [ys]
            if dst_ys is None:
                dst_ys = [[] for _ in ys]
            for dst_y, y in zip(dst_ys, ys): dst_y.append(y.cpu())
            if isinstance(t, dict):
                t = t['label']
            dst_t.append(t.cpu())
        dst_ys = [torch.cat(dst_y, dim=0).numpy() for dst_y in dst_ys]
        dst_t  =  torch.cat(dst_t, dim=0).numpy()

    return dst_ys, dst_t



# def main(i_fold, dir_result:str, path_predict:int, device:str) -> None:
def main(i_fold, dir_result:str, device:str) -> None:
    device = torch.device(device)
    torch.cuda.set_device(device)

    pathes = parse_path(dir_result)
    if pathes is None:
        return
    path_config, path_model, path_result = pathes

    config = load_json(path_config)

    pre, _, _, _ = models.get_process(device=device, **config['process'])

    pathes = load_json(config['dataset']['path_src'])
    # loader = dataset.get_dataset_single(pathes=pathes, train=False, shuffle=False, **config['dataset'])
    loader_train, loader_eval_tr, loader_eval_vl = \
        dataset.get_dataset(i_fold=i_fold, shuffle=False, **config['dataset'])

    model = models.get_model(**config['model'])
    model.load_state_dict(torch.load(path_model, map_location='cpu'))
    model = model.to(device)
    ys, t = predict(loader_eval_vl, model, device, pre)
    p = loader_eval_vl.dataset.pathes

    dst = [ys[0], t, p]

    utils.makedirs(os.path.join(path_result, str(i_fold), 'predict_val'))
    save_pickle(os.path.join(path_result, str(i_fold), 'predict_val')+'.pickle', dst)

    save_pickle(os.path.join(path_result, str(i_fold), 'predict_val')+'_appendix.pickle', ys[1:])

    path_result = pathlib.Path(path_result)
    utils.command_log(path_result.parents[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--target', '-t', required=True, type=str)
    parser.add_argument('--device', '-d', required=True, type=str)
    parser.add_argument('--fold',   '-f', required=False, type=str, nargs='*',
                                          default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    for i_fold in tqdm.tqdm(args.fold, leave=False):
        dir_fold = os.path.join('../', str(i_fold))
        # main(i_fold, dir_fold, args.target, args.device)
        main(i_fold, dir_fold, args.device)
