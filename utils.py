import json
import os
import pickle
import csv

import numpy as np
from torch import Tensor
from scipy import stats
import sys
import datetime


def mkdir(path):
    if not os.path.exists(path) or not os.path.isdir(path):
        os.mkdir(path)
        return True
    return False

def makedirs(path):
    dirname = os.path.dirname(path)
    if dirname != '' and not os.path.exists(dirname):
        os.makedirs(dirname)


def argmax(l:list):
    if isinstance(l, np.ndarray):
        return l.argmax()
    return l.index(max(l))


def mode(arr):
    return int(stats.mode(arr)[0])


def load_json(path):
    with open(path, 'r', encoding='utf_8') as f:
        dst = json.load(f)
    return dst

def save_json(path, obj):
    with open(path, 'w', encoding='utf_8') as f:
        json.dump(obj, f, indent=4)
    print(f'output: {path}')


def load_csv(path, encoding='utf_8_sig'):
    with open(path, 'r', newline='', encoding=encoding) as f:
        reader = csv.reader(f, delimiter=',')
        dst = list(reader)
    return dst


def save_csv(path, obj, encoding='utf_8_sig', mode='w'):
    makedirs(path)
    with open(path, mode, newline='', encoding=encoding) as f:
        wtr = csv.writer(f)
        wtr.writerows(obj)
    print(f'output: {path}')


def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'output: {path}')

def load_text(path, encoding='utf_8_sig'):
    with open(path, 'r', encoding=encoding) as f:
        text = f.read()
    return text


def save_text(path, text, encoding='utf_8_sig', mode='w'):
    makedirs(path)
    with open(path, mode, encoding=encoding) as f:
        f.write(text)
    print(f'output: {path}')



def as_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.array(x)
    if x.requires_grad:
        x = x.detach()
    if x.device.type == 'cuda':
        x = x.cpu()

    return x.numpy()


def TopN_accuracy(y, t, n):
    if isinstance(y, Tensor):
        y = as_numpy(y)
    if isinstance(t, Tensor):
        t = as_numpy(t)
    if isinstance(y, list):
        y = np.array(y)
    if isinstance(t, list):
        t = np.array(t)

    arg = np.argsort(y, axis=1)
    top5 = arg[:, -n:]

    contain = (top5 - t.reshape(-1,1)) == 0

    return float(np.sum(contain) / y.shape[0])


def macro_sensitivity(y_pred, y_true, n_classes=2):
    """
    マクロ平均センシティビティ（各クラスのRecall/TPRの平均）を計算
    """
    if isinstance(y_pred, Tensor):
        y_pred = as_numpy(y_pred)
    if isinstance(y_true, Tensor):
        y_true = as_numpy(y_true)
    
    # 予測値を確率からクラス予測に変換
    if y_pred.ndim > 1:
        y_pred_class = np.argmax(y_pred, axis=1)
    else:
        y_pred_class = y_pred
    
    sensitivities = []
    
    for class_idx in range(n_classes):
        # 各クラスについてTrue Positive, False Negativeを計算
        tp = np.sum((y_true == class_idx) & (y_pred_class == class_idx))
        fn = np.sum((y_true == class_idx) & (y_pred_class != class_idx))
        
        # Sensitivity = TP / (TP + FN)
        if tp + fn > 0:
            sensitivity = tp / (tp + fn)
        else:
            sensitivity = 0.0
        
        sensitivities.append(sensitivity)
    
    # マクロ平均
    return np.mean(sensitivities)


def convert_level(preds, mat, truthes=None, cnv=None, level=None):
    if isinstance(mat, str):
        mat = load_json(mat)[level]
        mat = np.array(mat)
    preds = np.array(preds)
    preds = (mat @ preds.T).T
    if truthes is None:
        return preds

    if isinstance(cnv, str):
        cnv = load_json(cnv)[level]
    truthes = [cnv[i] for i in truthes]
    return preds, truthes


def command_log(dir):
    output = []
    argv = sys.argv
    argv = ' '.join(argv)

    date = datetime.datetime.today() \
           .astimezone(datetime.timezone(datetime.timedelta(hours=9))) \
           .strftime('%Y_%m_%d__%H_%M_%S')
    output.append(date)

    output.append(os.getcwd())
    output.append('python3 ' + argv)
    output.append('\n')

    output = '\n'.join(output)

    if not os.path.isdir(dir):
        dir = os.path.dirname(dir)
    path = os.path.join(dir, 'command_log.txt')
    save_text(path, output, mode='a')
    save_text("./command_log.txt", output, mode='a')


def normalize_labels_to_binary():
    """ラベルを2クラス（0,1）に正規化する"""
    files = ['train_metadata_standardized.json', 'validation_metadata_standardized.json']
    
    for file_name in files:
        with open(file_name, 'r') as f:
            data = json.load(f)
        
        for entry in data:
            if 'LABEL' in entry:
                # 31 -> 1, 0 -> 0 に変換
                entry['LABEL'] = 1 if entry['LABEL'] == 31 else 0
                # CASEも更新
                if 'CASE' in entry:
                    case_parts = entry['CASE'].split('___')
                    if len(case_parts) >= 3:
                        case_parts[-1] = str(entry['LABEL'])
                        entry['CASE'] = '___'.join(case_parts)
        
        output_file = file_name.replace('.json', '_binary.json')
        save_json(output_file, data)  # 既存のsave_json関数を使用
        
        print(f"Created {output_file} with binary labels")
