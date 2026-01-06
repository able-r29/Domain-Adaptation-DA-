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
    """モデルディレクトリからパスを解析（DANN対応版）"""
    if not os.path.exists(dir_result):
        print(f"❌ ディレクトリが存在しません: {dir_result}")
        return None

    path_config = os.path.join(dir_result, 'config.json')
    if not os.path.exists(path_config):
        print(f"❌ config.jsonが見つかりません: {path_config}")
        return None
    
    # .pt と .pth の両方を検索
    pathes_model_pt = glob.glob(os.path.join(dir_result, '*.pt'))
    pathes_model_pth = glob.glob(os.path.join(dir_result, '*.pth'))
    pathes_model = pathes_model_pt + pathes_model_pth
    
    # ベストモデルを優先的に選択
    best_model = None
    for path in pathes_model:
        if 'best_model' in os.path.basename(path):
            best_model = path
            break
    
    if best_model is None and len(pathes_model) > 0:
        pathes_model.sort()
        best_model = pathes_model[-1]
    
    if best_model is None:
        print(f"❌ モデルファイルが見つかりません")
        print(f"   検索パス: {dir_result}/*.pt, {dir_result}/*.pth")
        print(f"   見つかったファイル:")
        for f in os.listdir(dir_result):
            print(f"     - {f}")
        return None
    
    print(f"✓ 使用するモデル: {best_model}")
    
    path_result = os.path.join(dir_result, 'predict')
    return path_config, best_model, path_result

def load_dann_model(path_model, config, device):
    """DANNモデルの読み込み（チェックポイント対応・改良版）"""
    checkpoint = torch.load(path_model, map_location='cpu')
    
    # チェックポイント形式かどうかを判定
    is_checkpoint = 'classifier_state_dict' in checkpoint or 'epoch' in checkpoint
    
    if is_checkpoint:
        print("✓ DANNチェックポイント形式のモデルを検出")
        
        # モデルを初期化
        sys.path.append('.')
        import models.models as models
        from domain_discriminator import DANNClassifier, DomainDiscriminator
        
        # バックボーンモデルの取得
        backbone = models.get_model(**config['model'])
        
        # DANN構成の取得
        dann_config = config.get('dann', {})
        num_classes = config['model'].get('n_class', 2)
        bottleneck_dim = dann_config.get('bottleneck_dim', 256)
        domain_hidden = dann_config.get('domain_hidden_size', 1024)
        
        # DANNモデルの構築
        classifier = DANNClassifier(backbone, num_classes, bottleneck_dim)
        domain_discriminator = DomainDiscriminator(bottleneck_dim, domain_hidden)
        
        # 状態辞書の読み込み
        classifier_state = checkpoint['classifier_state_dict']
        domain_state = checkpoint['domain_discriminator_state_dict']
        
        # ★ DataParallelのstate_dictから'module.'プレフィックスを除去
        if any(k.startswith('module.') for k in classifier_state.keys()):
            classifier_state = {k.replace('module.', ''): v for k, v in classifier_state.items()}
        if any(k.startswith('module.') for k in domain_state.keys()):
            domain_state = {k.replace('module.', ''): v for k, v in domain_state.items()}
        
        classifier.load_state_dict(classifier_state)
        domain_discriminator.load_state_dict(domain_state)
        
        print(f"  エポック: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Val AUC: {checkpoint.get('best_val_auc', 'unknown')}")
        
        # 予測用にclassifierのみを返す
        model = classifier
    else:
        print("✓ 通常のstate_dict形式のモデルを検出")
        sys.path.append('.')
        import models.models as models
        
        model = models.get_model(**config['model'])
        
        # ★ 修正: checkpointがstate_dictかどうかを判定
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            # チェックポイント内にstate_dictがある場合
            state_dict = checkpoint['state_dict']
        elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
            # checkpoint自体がstate_dictの場合
            state_dict = checkpoint
        else:
            raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
        
        # DataParallelのプレフィックスを除去
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
    
    return model.to(device)

def predict(loader, model, device, preprocess):
    """予測実行（DANN対応）"""
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        dst_ys = None
        dst_t = []
        
        for batch in tqdm.tqdm(loader, total=len(loader), leave=False):
            x, t = preprocess(batch, device, True)
            
            # DANN対応：predictメソッドがない場合はforwardを使用
            if hasattr(model, 'predict'):
                ys = model.predict(x)
            else:
                # DANNClassifierの場合
                logits, _ = model(x)
                ys = logits
            
            if isinstance(ys, torch.Tensor):
                ys = [ys]
            if dst_ys is None:
                dst_ys = [[] for _ in ys]
            for dst_y, y in zip(dst_ys, ys): 
                dst_y.append(y.cpu())
            if isinstance(t, dict):
                t = t['label']
            dst_t.append(t.cpu())
        
        dst_ys = [torch.cat(dst_y, dim=0).numpy() for dst_y in dst_ys]
        dst_t = torch.cat(dst_t, dim=0).numpy()

    return dst_ys, dst_t

def plot_confusion_matrix(true_labels, predicted_classes, save_path, class_names=None):
    """Confusion Matrixを作成・保存"""
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
    """ROC曲線を作成・保存"""
    from sklearn.metrics import roc_curve as sklearn_roc_curve, auc as sklearn_auc
    
    def manual_softmax(logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    normalized_predictions = manual_softmax(predictions)
    
    plt.figure(figsize=(8, 6))
    
    if predictions.shape[1] == 2:
        fpr, tpr, _ = sklearn_roc_curve(true_labels, normalized_predictions[:, 1])
        roc_auc = sklearn_auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
    else:
        from sklearn.preprocessing import label_binarize
        from itertools import cycle
        
        n_classes = predictions.shape[1]
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]
            
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
    """詳細な予測結果をCSVファイルとして保存"""
    def manual_softmax(logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    normalized_predictions = manual_softmax(predictions)
    
    predicted_classes = np.argmax(normalized_predictions, axis=1)
    max_probabilities = np.max(normalized_predictions, axis=1)
    correct_predictions = (true_labels == predicted_classes)
    
    data = []
    for i in range(len(file_paths)):
        if isinstance(file_paths[i], dict):
            filename = file_paths[i].get('filename', f'unknown_{i}')
            image_path = file_paths[i].get('jpg_src', filename)
        else:
            image_path = str(file_paths[i])
            filename = os.path.basename(image_path)
        
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
    """Accuracy, AUC, Macro-Sensitivityを計算"""
    def manual_softmax(logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    normalized_predictions = manual_softmax(predictions)
    predicted_classes = np.argmax(normalized_predictions, axis=1)
    
    accuracy = accuracy_score(true_labels, predicted_classes)
    
    try:
        if predictions.shape[1] == 2:
            pred_probabilities = normalized_predictions[:, 1]
            auc_score = roc_auc_score(true_labels, pred_probabilities)
        else:
            auc_score = roc_auc_score(true_labels, normalized_predictions, multi_class='ovr')
    except ValueError as e:
        print(f"AUC計算エラー: {e}")
        auc_score = 0.5
    
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

def main(dir_result:str, path_predict:str, device:str) -> None:
    device = torch.device(device)
    
    print(f"\n{'='*60}")
    print(f"DANN Prediction Script")
    print(f"{'='*60}")
    print(f"モデルディレクトリ: {dir_result}")
    print(f"予測対象データ: {path_predict}")
    print(f"使用デバイス: {device}")
    print(f"{'='*60}\n")
    
    pathes = parse_path(dir_result)
    if pathes is None:
        print("\n❌ 処理を中断します")
        return
    path_config, path_model, path_result = pathes

    config = load_json(path_config)
    print(f"\n✓ 設定ファイル読み込み完了")

    sys.path.append('.')
    import models.models as models
    import datasets.datasets as dataset
    
    pre, _, _, _ = models.get_process(device=device, **config['process'])

    pathes_data = load_json(path_predict)
    loader = dataset.get_dataset_single(
        pathes=pathes_data, 
        train=False, 
        shuffle=False, 
        **config['dataset']
    )
    print(f"✓ データセット読み込み完了: {len(loader.dataset)}件")

    print(f"\n{'='*60}")
    print("モデル読み込み中...")
    print(f"{'='*60}")
    model = load_dann_model(path_model, config, device)
    print(f"✓ モデル読み込み完了\n")

    print(f"{'='*60}")
    print("予測実行中...")
    print(f"{'='*60}")
    ys, t = predict(loader, model, device, pre)
    p = loader.dataset.pathes

    print(f"\n{'='*60}")
    print("予測値の詳細確認")
    print(f"{'='*60}")
    print(f"予測値形状: {ys[0].shape}")
    print(f"予測値の統計情報:")
    print(f"  最小値: {np.min(ys[0]):.6f}")
    print(f"  最大値: {np.max(ys[0]):.6f}")
    print(f"  平均値: {np.mean(ys[0]):.6f}")
    
    print(f"\n最初の5サンプルの生の予測値:")
    for i in range(min(5, len(ys[0]))):
        raw_probs = ys[0][i]
        sum_probs = np.sum(raw_probs)
        print(f"  サンプル{i}: {[f'{v:.6f}' for v in raw_probs]} (合計: {sum_probs:.6f})")
    
    def manual_softmax(logits):
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    softmax_probs = manual_softmax(ys[0])
    print(f"\nSoftmax適用後の最初の5サンプル:")
    for i in range(min(5, len(softmax_probs))):
        soft_probs = softmax_probs[i]
        sum_probs = np.sum(soft_probs)
        print(f"  サンプル{i}: {[f'{v:.6f}' for v in soft_probs]} (合計: {sum_probs:.6f})")
    
    print(f"\n{'='*60}")
    print("メトリクス計算中...")
    print(f"{'='*60}")
    metrics = calculate_metrics(ys[0], t)
    predicted_classes = np.argmax(ys[0], axis=1)
    
    print(f"\n{'='*60}")
    print("予測結果メトリクス")
    print(f"{'='*60}")
    print(f"データ件数: {len(t)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Macro-Sensitivity: {metrics['macro_sensitivity']:.4f}")
    
    unique, counts = np.unique(t, return_counts=True)
    print(f"\nクラス分布（正解ラベル）:")
    for cls, count in zip(unique, counts):
        print(f"  クラス {cls}: {count}件")
    
    pred_unique, pred_counts = np.unique(predicted_classes, return_counts=True)
    print(f"\nクラス分布（予測）:")
    for cls, count in zip(pred_unique, pred_counts):
        print(f"  クラス {cls}: {count}件")
    
    correct_predictions = (t == predicted_classes)
    print(f"\n正解数: {np.sum(correct_predictions)}/{len(t)}")
    print(f"誤予測数: {len(t) - np.sum(correct_predictions)}/{len(t)}")

    print(f"\n{'='*60}")
    print("詳細分析結果生成中...")
    print(f"{'='*60}")
    
    print("\n1. Confusion Matrix作成中...")
    cm = plot_confusion_matrix(t, predicted_classes, path_result+'_confusion_matrix.png')
    print(f"   ✓ 保存: {path_result}_confusion_matrix.png")
    
    print("\n2. ROC曲線作成中...")
    plot_roc_curve(t, ys[0], path_result+'_roc_curve.png')
    print(f"   ✓ 保存: {path_result}_roc_curve.png")
    
    print("\n3. 詳細CSVファイル作成中...")
    df = create_detailed_csv(p, t, ys[0], path_result+'_detailed_results.csv')
    print(f"   ✓ 保存: {path_result}_detailed_results.csv")
    print(f"   CSVファイル形状: {df.shape}")

    dst = [ys[0], t, p]
    save_pickle(path_result+'.pickle', dst)
    save_pickle(path_result+'_metrics.pickle', metrics)
    save_pickle(path_result+'_appendix.pickle', ys[1:])

    metrics_json = {k: float(v) for k, v in metrics.items()}
    with open(path_result+'_metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    np.savetxt(path_result+'_confusion_matrix.txt', cm, fmt='%d')

    print(f"\n{'='*60}")
    print("出力ファイル一覧")
    print(f"{'='*60}")
    output_files = [
        (path_result+'.pickle', '予測結果(Pickle)'),
        (path_result+'_metrics.pickle', 'メトリクス(Pickle)'),
        (path_result+'_metrics.json', 'メトリクス(JSON)'),
        (path_result+'_appendix.pickle', '追加情報'),
        (path_result+'_confusion_matrix.png', 'Confusion Matrix(画像)'),
        (path_result+'_confusion_matrix.txt', 'Confusion Matrix(テキスト)'),
        (path_result+'_roc_curve.png', 'ROC曲線'),
        (path_result+'_detailed_results.csv', '詳細結果CSV')
    ]
    
    for file_path, description in output_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✓ {description:30s} {os.path.basename(file_path):40s} ({file_size:,} bytes)")
        else:
            print(f"✗ {description:30s} {os.path.basename(file_path):40s}")

    print(f"\n{'='*60}")
    print("✓ 予測処理完了")
    print(f"{'='*60}\n")

    path_result = pathlib.Path(path_result)
    utils.command_log(path_result.parents[1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DANN Model Prediction Script',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--target', '-t', required=True, type=str,
                       help='予測対象データのパスを含むJSONファイル')
    parser.add_argument('--device', '-d', required=True, type=str,
                       help='使用デバイス（例: cuda:0, cpu）')
    parser.add_argument('--model_dir', '-m', required=True, type=str, 
                       help='モデルディレクトリ（.pthファイルとconfig.jsonを含む）')

    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')
    main(args.model_dir, args.target, args.device)