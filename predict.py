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
from sklearn.metrics import accuracy_score, roc_auc_score
import utils  # macro_sensitivity関数が含まれていると仮定

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

    path_result = os.path.join(dir_result, 'predict')

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



def calculate_metrics(predictions, true_labels):
    """
    Accuracy, AUC, Macro-Sensitivityを計算
    """
    # 予測クラス（最大確率のインデックス）
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_classes)
    
    # AUC (2クラス分類を想定)
    try:
        if predictions.shape[1] == 2:
            # 2クラス分類: クラス1の確率を使用
            pred_probabilities = predictions[:, 1]
            auc = roc_auc_score(true_labels, pred_probabilities)
        else:
            # 多クラス分類: OvR (One-vs-Rest) AUC
            auc = roc_auc_score(true_labels, predictions, multi_class='ovr')
    except ValueError as e:
        print(f"AUC計算エラー: {e}")
        auc = 0.5  # デフォルト値
    
    # Macro-Sensitivity
    try:
        macro_sens = utils.macro_sensitivity(predictions, true_labels, n_classes=predictions.shape[1])
    except Exception as e:
        print(f"Macro-Sensitivity計算エラー: {e}")
        macro_sens = 0.0  # デフォルト値
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'macro_sensitivity': macro_sens
    }

def main(dir_result:str, path_predict:int, device:str) -> None:
    device = torch.device(device)
    torch.cuda.set_device(device)

    pathes = parse_path(dir_result)
    if pathes is None:
        return
    path_config, path_model, path_result = pathes

    config = load_json(path_config)

    # 正しいインポート方法
    sys.path.append('.')  # カレントディレクトリをパスに追加
    import models.models as models
    import datasets.datasets as dataset
    
    pre, _, _, _ = models.get_process(device=device, **config['process'])

    pathes = load_json(path_predict)
    
    # get_dataset_single関数を直接呼び出し
    loader = dataset.get_dataset_single(pathes=pathes, train=False, shuffle=False, **config['dataset'])

    model = models.get_model(**config['model'])
    model.load_state_dict(torch.load(path_model, map_location='cpu'))
    model = model.to(device)
    ys, t = predict(loader, model, device, pre)
    p = loader.dataset.pathes

    # メトリクス計算
    metrics = calculate_metrics(ys[0], t)
    
    # 結果を表示
    print(f"\n=== 予測結果メトリクス ===")
    print(f"データ件数: {len(t)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Macro-Sensitivity: {metrics['macro_sensitivity']:.4f}")
    
    # クラス分布を表示
    unique, counts = np.unique(t, return_counts=True)
    print(f"\nクラス分布:")
    for cls, count in zip(unique, counts):
        print(f"  クラス {cls}: {count}件")

    dst = [ys[0], t, p]
    save_pickle(path_result+'.pickle', dst)
    
    # メトリクスも保存
    save_pickle(path_result+'_metrics.pickle', metrics)
    save_pickle(path_result+'_appendix.pickle', ys[1:])

    # メトリクスをJSONでも保存（人間が読みやすい形式）
    metrics_json = {k: float(v) for k, v in metrics.items()}  # numpy型をfloatに変換
    with open(path_result+'_metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)

    path_result = pathlib.Path(path_result)
    utils.command_log(path_result.parents[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', required=True, type=str)
    parser.add_argument('--device', '-d', required=True, type=str)
    parser.add_argument('--model_dir', '-m', required=True, type=str, 
                        help='Path to model directory containing .pt file and config.json')

    #parser.add_argument('--fold',   '-f', required=False, type=str, nargs='*', default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    #for i_fold in tqdm.tqdm(args.fold, leave=False):
    #    dir_fold = os.path.join('../', str(i_fold))
    #    main(dir_fold, args.target, args.device)
    main(args.model_dir, args.target, args.device)