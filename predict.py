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
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

def plot_confusion_matrix(true_labels, predicted_classes, save_path, class_names=None):
    """
    Confusion Matrixを作成・保存
    """
    cm = confusion_matrix(true_labels, predicted_classes)
    
    plt.figure(figsize=(8, 6))
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm

def plot_roc_curve(true_labels, predictions, save_path, class_names=None):
    """
    ROC曲線を作成・保存
    """
    from sklearn.metrics import roc_curve as sklearn_roc_curve, auc as sklearn_auc
    
    # Softmaxを適用して確率に変換
    def manual_softmax(logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    normalized_predictions = manual_softmax(predictions)
    
    plt.figure(figsize=(8, 6))
    
    if predictions.shape[1] == 2:
        # 2クラス分類の場合 - 正規化後の確率を使用
        fpr, tpr, _ = sklearn_roc_curve(true_labels, normalized_predictions[:, 1])
        roc_auc = sklearn_auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
    else:
        # 多クラス分類の場合（One-vs-Rest）
        from sklearn.preprocessing import label_binarize
        from itertools import cycle
        
        n_classes = predictions.shape[1]
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
            
        # ラベルを二進化
        y_bin = label_binarize(true_labels, classes=list(range(n_classes)))
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
        for i, color in zip(range(n_classes), colors):
            fpr, tpr, _ = sklearn_roc_curve(y_bin[:, i], predictions[:, i])
            roc_auc = sklearn_auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_detailed_csv(file_paths, true_labels, predictions, save_path):
    """
    詳細な予測結果をCSVファイルとして保存
    """
    # Softmaxを適用して確率に正規化
    def manual_softmax(logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Logitsを確率に変換
    normalized_predictions = manual_softmax(predictions)
    
    predicted_classes = np.argmax(normalized_predictions, axis=1)
    max_probabilities = np.max(normalized_predictions, axis=1)  # 正規化後の最大確率
    correct_predictions = (true_labels == predicted_classes)
    
    # データフレームを作成
    data = []
    for i in range(len(file_paths)):
        # ファイルパスの形式を確認・処理
        if isinstance(file_paths[i], dict):
            filename = file_paths[i].get('filename', f'unknown_{i}')
            image_path = file_paths[i].get('jpg_src', filename)
        else:
            image_path = str(file_paths[i])
            filename = os.path.basename(image_path)
        
        # 正規化された確率を使用（重要な変更点）
        probs = {f'prob_class_{j}': normalized_predictions[i, j] for j in range(predictions.shape[1])}
        
        row = {
            'filename': filename,
            'image_path': image_path,
            'true_class': int(true_labels[i]),
            'predicted_class': int(predicted_classes[i]),
            'confidence': float(max_probabilities[i]),
            'correct': 'correct' if correct_predictions[i] else 'incorrect',
            **probs
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    
    return df

def calculate_metrics(predictions, true_labels):
    """
    Accuracy, AUC, Macro-Sensitivityを計算
    """
    # Softmaxを適用
    def manual_softmax(logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    normalized_predictions = manual_softmax(predictions)
    predicted_classes = np.argmax(normalized_predictions, axis=1)
    
    # Accuracy
    accuracy = accuracy_score(true_labels, predicted_classes)
    
    # AUC - 正規化後の確率を使用
    try:
        if predictions.shape[1] == 2:
            pred_probabilities = normalized_predictions[:, 1]  # ← 修正
            auc_score = roc_auc_score(true_labels, pred_probabilities)
        else:
            auc_score = roc_auc_score(true_labels, normalized_predictions, multi_class='ovr')
    except ValueError as e:
        print(f"AUC計算エラー: {e}")
        auc_score = 0.5
    
    # Macro-Sensitivity - 正規化後の予測を使用
    try:
        macro_sens = utils.macro_sensitivity(normalized_predictions, true_labels, n_classes=predictions.shape[1])
    except Exception as e:
        print(f"Macro-Sensitivity計算エラー: {e}")
        macro_sens = 0.0
    
    return {
        'accuracy': accuracy,
        'auc': auc_score,
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

    # === 予測値の詳細確認（ここに追加） ===
    print(f"\n=== 予測値の詳細確認 ===")
    print(f"predictions形状: {ys[0].shape}")
    print(f"予測値の統計情報:")
    print(f"  最小値: {np.min(ys[0]):.6f}")
    print(f"  最大値: {np.max(ys[0]):.6f}")
    print(f"  平均値: {np.mean(ys[0]):.6f}")
    
    print(f"\n最初の5サンプルの生の予測値:")
    for i in range(min(5, len(ys[0]))):
        raw_probs = ys[0][i]
        sum_probs = np.sum(raw_probs)
        print(f"  サンプル{i}: [{raw_probs[0]:.6f}, {raw_probs[1]:.6f}] (合計: {sum_probs:.6f})")
    
    # Softmaxを適用した場合の確認
    def manual_softmax(logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    softmax_probs = manual_softmax(ys[0])
    print(f"\nSoftmax適用後の最初の5サンプル:")
    for i in range(min(5, len(softmax_probs))):
        soft_probs = softmax_probs[i]
        sum_probs = np.sum(soft_probs)
        print(f"  サンプル{i}: [{soft_probs[0]:.6f}, {soft_probs[1]:.6f}] (合計: {sum_probs:.6f})")
    
    # メトリクス計算
    metrics = calculate_metrics(ys[0], t)
    
    # 予測クラスを取得
    predicted_classes = np.argmax(ys[0], axis=1)
    
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
    
    # 予測精度の詳細
    correct_predictions = (t == predicted_classes)
    print(f"\n正解数: {np.sum(correct_predictions)}/{len(t)}")
    print(f"誤予測数: {len(t) - np.sum(correct_predictions)}/{len(t)}")

    # === 新機能: 詳細な分析結果の生成 ===
    
    # 1. Confusion Matrix
    print("\n=== Confusion Matrix作成中 ===")
    cm = plot_confusion_matrix(t, predicted_classes, path_result+'_confusion_matrix.png')
    print(f"Confusion Matrix保存: {path_result+'_confusion_matrix.png'}")
    
    # 2. ROC曲線
    print("\n=== ROC曲線作成中 ===")
    plot_roc_curve(t, ys[0], path_result+'_roc_curve.png')
    print(f"ROC曲線保存: {path_result+'_roc_curve.png'}")
    
    # 3. 詳細CSVファイル
    print("\n=== 詳細CSVファイル作成中 ===")
    df = create_detailed_csv(p, t, ys[0], path_result+'_detailed_results.csv')
    print(f"詳細結果CSV保存: {path_result+'_detailed_results.csv'}")
    print(f"CSVファイル形状: {df.shape}")

    # 既存の保存処理
    dst = [ys[0], t, p]
    save_pickle(path_result+'.pickle', dst)
    
    # メトリクスも保存
    save_pickle(path_result+'_metrics.pickle', metrics)
    save_pickle(path_result+'_appendix.pickle', ys[1:])

    # メトリクスをJSONでも保存（人間が読みやすい形式）
    metrics_json = {k: float(v) for k, v in metrics.items()}  # numpy型をfloatに変換
    with open(path_result+'_metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Confusion Matrixもテキストで保存
    np.savetxt(path_result+'_confusion_matrix.txt', cm, fmt='%d')

    print(f"\n=== 出力ファイル一覧 ===")
    output_files = [
        path_result+'.pickle',
        path_result+'_metrics.pickle',
        path_result+'_metrics.json',
        path_result+'_appendix.pickle',
        path_result+'_confusion_matrix.png',
        path_result+'_confusion_matrix.txt',
        path_result+'_roc_curve.png',
        path_result+'_detailed_results.csv'
    ]
    for file_path in output_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")

    path_result = pathlib.Path(path_result)
    utils.command_log(path_result.parents[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', '-t', required=True, type=str)
    parser.add_argument('--device', '-d', required=True, type=str)
    parser.add_argument('--model_dir', '-m', required=True, type=str, 
                        help='Path to model directory containing .pt file and config.json')

    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    main(args.model_dir, args.target, args.device)