#!/usr/bin/env python3
"""
Hanley McNeil法によるAUC検定プログラム
2つのモデルのAUC値の有意差を検定します
"""

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# 👇 ここでファイルパスを直接指定してください
# ==========================================
DEFAULT_PATHS = {
    # 基準となるモデル（固定）
    'reference': '../resnet18_before_gamma4/before_test1/predict_detailed_results.csv',
    'reference_name': 'before_test1',
    
    # 比較対象のモデル（必要に応じて .csv に変更）
    'models': [
        {'path': '../resnet18_before_gamma4/all_after_test1/predict_detailed_results.csv', 'name': 'all_after_test1'},
        {'path': '../resnet18_before_gamma4/KMOface/predict_detailed_results.csv', 'name': 'KMOface'},
        {'path': '../resnet18_before_gamma4/KMUbody/predict_detailed_results.csv', 'name': 'KMUbody'},
        {'path': '../resnet18_before_gamma4/KSObody/predict_detailed_results.csv', 'name': 'KSObody'},
        {'path': '../resnet18_before_gamma4/SSObody/predict_detailed_results.csv', 'name': 'SSObody'},
        {'path': '../resnet18_before_gamma4/SSOface/predict_detailed_results.csv', 'name': 'SSOface'},
        {'path': '../resnet18_before_gamma4/SSUbody/predict_detailed_results.csv', 'name': 'SSUbody'},
        {'path': '../resnet18_before_gamma4/YNObody/predict_detailed_results.csv', 'name': 'YNObody'},
    ],
    
    'output': './hanley_mcneil_multi_results',
    'alpha': 0.05,
    'bootstrap': True
}

def load_prediction_results(file_path):
    """予測結果ファイルを読み込み（pickle, CSV, NPZ対応）"""
    print(f"Loading: {file_path}")
    
    file_path = Path(file_path)
    
    # ファイル存在チェック
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix.lower() == '.csv':
        # CSV形式の読み込み
        df = pd.read_csv(file_path)
        print(f"  CSV shape: {df.shape}")
        print(f"  CSV columns: {list(df.columns)}")
        
        # 真のラベルを特定（true_classを優先的に使用）
        if 'true_class' in df.columns:
            true_labels = df['true_class'].values
            print(f"  Using label column: true_class")
        else:
            # フォールバック
            label_columns = [col for col in df.columns if col.lower() in 
                            ['true_label', 'label', 'ground_truth', 'gt', 'y_true', 'actual']]
            if not label_columns:
                raise ValueError(f"No label column found. Available columns: {list(df.columns)}")
            true_labels = df[label_columns[0]].values
            print(f"  Using label column: {label_columns[0]}")
        
        # 確率列を特定（prob_class_0, prob_class_1を優先的に使用）
        prob_columns = [col for col in df.columns if col.startswith('prob_class_')]
        
        if len(prob_columns) >= 2:
            # prob_class_0, prob_class_1が存在する場合
            prob_columns_sorted = sorted(prob_columns)  # prob_class_0, prob_class_1の順序を保証
            predictions = df[prob_columns_sorted].values
            print(f"  Using probability columns: {prob_columns_sorted}")
            print(f"  ✓ These are already normalized probabilities from CSV")
        else:
            # フォールバック：他の予測列を探す
            pred_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                           ['pred', 'prob', 'score', 'logit'])]
            
            if len(pred_columns) >= 2:
                predictions = df[pred_columns].values
                print(f"  Using prediction columns: {pred_columns}")
            elif len(pred_columns) == 1:
                predictions = df[pred_columns[0]].values.reshape(-1, 1)
                print(f"  Using single prediction column: {pred_columns[0]}")
            else:
                raise ValueError("No prediction columns found")
        
        file_paths = df.get('file_path', None)
        
        # 確率値の検証
        if predictions.shape[1] == 2:
            row_sums = np.sum(predictions, axis=1)
            print(f"  First 5 probability pairs: {predictions[:5]}")
            print(f"  Row sums (first 5): {row_sums[:5]}")
            print(f"  Min probability: {np.min(predictions):.6f}")
            print(f"  Max probability: {np.max(predictions):.6f}")
        
    elif file_path.suffix.lower() in ['.pkl', '.pickle']:
        # Pickle形式の読み込み
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"  Pickle content type: {type(data)}")
        
        if isinstance(data, tuple) and len(data) >= 2:
            predictions = data[0]
            true_labels = data[1]
            file_paths = data[2] if len(data) > 2 else None
            
        elif isinstance(data, list):
            print(f"  List length: {len(data)}")
            if len(data) > 0:
                print(f"  First element type: {type(data[0])}")
                
                if isinstance(data[0], dict):
                    predictions = np.array([item.get('prediction', item.get('pred', 0)) for item in data])
                    true_labels = np.array([item.get('true_label', item.get('label', 0)) for item in data])
                    file_paths = [item.get('file_path', f'sample_{i}') for i in range(len(data))]
                    
                elif isinstance(data[0], (list, tuple)) and len(data[0]) >= 2:
                    predictions = np.array([item[0] for item in data])
                    true_labels = np.array([item[1] for item in data])
                    file_paths = [item[2] if len(item) > 2 else f'sample_{i}' for i in range(len(data))]
                    
                else:
                    raise ValueError(f"Unsupported list content type: {type(data[0])}")
            else:
                raise ValueError("Empty list in pickle file")
                
        elif isinstance(data, dict):
            print(f"  Dictionary keys: {list(data.keys())}")
            predictions = data.get('predictions', data.get('pred', data.get('scores')))
            true_labels = data.get('true_labels', data.get('labels', data.get('y_true')))
            file_paths = data.get('file_paths', data.get('files'))
            
            if predictions is None or true_labels is None:
                raise ValueError(f"Required keys not found. Available: {list(data.keys())}")
                
        else:
            raise ValueError(f"Unexpected pickle format: {type(data)}")
    
    elif file_path.suffix.lower() == '.npz':
        # NPZ形式の読み込み
        data = np.load(file_path, allow_pickle=True)
        print(f"  NPZ arrays: {list(data.keys())}")
        
        predictions = data.get('predictions', data.get('pred', data.get('scores')))
        true_labels = data.get('true_labels', data.get('labels', data.get('y_true')))
        file_paths = data.get('file_paths', data.get('files', None))
        
        if predictions is None or true_labels is None:
            raise ValueError(f"Required arrays not found. Available: {list(data.keys())}")
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # NumPy配列に変換
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  True labels shape: {true_labels.shape}")
    print(f"  Positive cases: {np.sum(true_labels == 1)}")
    print(f"  Negative cases: {np.sum(true_labels == 0)}")
    print(f"  Unique labels: {np.unique(true_labels)}")
    
    # バリデーション
    if len(predictions) != len(true_labels):
        raise ValueError(f"Predictions and labels length mismatch: {len(predictions)} vs {len(true_labels)}")
    
    return predictions, true_labels, file_paths

def softmax(logits):
    """Softmax関数"""
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def calculate_auc_variance(y_true, y_scores):
    """Hanley McNeil法によるAUCの分散計算"""
    auc_value = roc_auc_score(y_true, y_scores)
    
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return auc_value, float('inf')
    
    Q1 = auc_value / (2 - auc_value)
    Q2 = (2 * auc_value ** 2) / (1 + auc_value)
    
    auc_variance = (auc_value * (1 - auc_value) + 
                   (n_pos - 1) * (Q1 - auc_value ** 2) + 
                   (n_neg - 1) * (Q2 - auc_value ** 2)) / (n_pos * n_neg)
    
    return auc_value, auc_variance

def hanley_mcneil_independent_test(y_true1, scores1, y_true2, scores2, alpha=0.05):
    """独立サンプルに対するHanley McNeil法による2つのAUCの比較検定"""
    auc1, var1 = calculate_auc_variance(y_true1, scores1)
    auc2, var2 = calculate_auc_variance(y_true2, scores2)
    
    correlation = 0.0
    var_diff = var1 + var2
    
    if var_diff <= 0:
        z_score = float('inf')
        p_value = 0.0
    else:
        z_score = (auc1 - auc2) / np.sqrt(var_diff)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    ci_lower = (auc1 - auc2) - stats.norm.ppf(1 - alpha/2) * np.sqrt(var_diff)
    ci_upper = (auc1 - auc2) + stats.norm.ppf(1 - alpha/2) * np.sqrt(var_diff)
    
    return {
        'auc1': auc1,
        'auc2': auc2,
        'auc_diff': auc1 - auc2,
        'var1': var1,
        'var2': var2,
        'correlation': correlation,
        'var_diff': var_diff,
        'z_score': z_score,
        'p_value': p_value,
        'alpha': alpha,
        'significant': p_value < alpha,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'better_performance': 'Reference' if auc1 > auc2 else 'Comparison' if auc2 > auc1 else 'Equal',
        'sample_size1': len(y_true1),
        'sample_size2': len(y_true2),
        'test_type': 'Independent samples Hanley-McNeil test'
    }

def bootstrap_auc_comparison_independent(y_true1, scores1, y_true2, scores2, n_bootstrap=1000, alpha=0.05):
    """独立サンプルに対するブートストラップ法によるAUC比較"""
    auc_diffs = []
    
    for _ in range(n_bootstrap):
        indices1 = np.random.choice(len(y_true1), size=len(y_true1), replace=True)
        indices2 = np.random.choice(len(y_true2), size=len(y_true2), replace=True)
        
        y_boot1 = y_true1[indices1]
        s_boot1 = scores1[indices1]
        y_boot2 = y_true2[indices2]
        s_boot2 = scores2[indices2]
        
        try:
            auc1_boot = roc_auc_score(y_boot1, s_boot1)
            auc2_boot = roc_auc_score(y_boot2, s_boot2)
            auc_diffs.append(auc1_boot - auc2_boot)
        except ValueError:
            continue
    
    auc_diffs = np.array(auc_diffs)
    
    ci_lower = np.percentile(auc_diffs, 100 * alpha/2)
    ci_upper = np.percentile(auc_diffs, 100 * (1 - alpha/2))
    
    observed_diff = roc_auc_score(y_true1, scores1) - roc_auc_score(y_true2, scores2)
    if observed_diff >= 0:
        p_value_boot = 2 * np.mean(auc_diffs <= 0)
    else:
        p_value_boot = 2 * np.mean(auc_diffs >= 0)
    
    return {
        'auc_diff_mean': np.mean(auc_diffs),
        'auc_diff_std': np.std(auc_diffs),
        'ci_lower_boot': ci_lower,
        'ci_upper_boot': ci_upper,
        'p_value_boot': p_value_boot,
        'significant_boot': not (ci_lower <= 0 <= ci_upper)
    }

def create_multi_comparison_plot(reference_data, comparison_data, save_path):
    """複数比較の結果をプロット作成"""
    ref_true, ref_scores, ref_name = reference_data
    
    n_comparisons = len(comparison_data)
    fig_height = max(10, 4 * n_comparisons)
    fig, axes = plt.subplots(n_comparisons, 3, figsize=(18, fig_height))
    
    if n_comparisons == 1:
        axes = axes.reshape(1, -1)
    
    ref_auc = roc_auc_score(ref_true, ref_scores)
    ref_fpr, ref_tpr, _ = roc_curve(ref_true, ref_scores)
    
    for i, (comp_true, comp_scores, comp_name) in enumerate(comparison_data):
        comp_auc = roc_auc_score(comp_true, comp_scores)
        comp_fpr, comp_tpr, _ = roc_curve(comp_true, comp_scores)
        
        # ROC曲線比較
        axes[i, 0].plot(ref_fpr, ref_tpr, 'b-', lw=2, 
                       label=f'{ref_name} (AUC = {ref_auc:.4f})')
        axes[i, 0].plot(comp_fpr, comp_tpr, 'r-', lw=2, 
                       label=f'{comp_name} (AUC = {comp_auc:.4f})')
        axes[i, 0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        axes[i, 0].set_xlim([0.0, 1.0])
        axes[i, 0].set_ylim([0.0, 1.05])
        axes[i, 0].set_xlabel('False Positive Rate')
        axes[i, 0].set_ylabel('True Positive Rate')
        axes[i, 0].set_title(f'ROC: {ref_name} vs {comp_name}')
        axes[i, 0].legend(loc="lower right")
        axes[i, 0].grid(True, alpha=0.3)
        
        # AUC棒グラフ
        models = [ref_name, comp_name]
        aucs = [ref_auc, comp_auc]
        colors = ['blue', 'red']
        
        bars = axes[i, 1].bar(models, aucs, color=colors, alpha=0.7)
        axes[i, 1].set_ylabel('AUC')
        axes[i, 1].set_title(f'AUC Comparison')
        axes[i, 1].set_ylim([0.0, 1.0])
        axes[i, 1].grid(True, alpha=0.3)
        
        for bar, auc in zip(bars, aucs):
            height = bar.get_height()
            axes[i, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{auc:.3f}', ha='center', va='bottom')
        
        # スコア分布比較
        axes[i, 2].hist(ref_scores[ref_true==1], bins=30, alpha=0.5, 
                       label=f'{ref_name} (Pos)', color='blue')
        axes[i, 2].hist(comp_scores[comp_true==1], bins=30, alpha=0.5, 
                       label=f'{comp_name} (Pos)', color='red')
        axes[i, 2].set_xlabel('Prediction Score')
        axes[i, 2].set_ylabel('Frequency')
        axes[i, 2].set_title(f'Score Distribution (Positive Cases)')
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_plot(results_summary, save_path):
    """全結果サマリープロット"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    models = [r['model_name'] for r in results_summary]
    aucs = [r['auc'] for r in results_summary]
    p_values = [r['p_value'] for r in results_summary]
    significant = [r['significant'] for r in results_summary]
    
    # AUC比較
    colors = ['blue' if i == 0 else ('red' if sig else 'gray') 
              for i, sig in enumerate(significant)]
    bars = ax1.bar(models, aucs, color=colors, alpha=0.7)
    ax1.set_ylabel('AUC')
    ax1.set_title('AUC Comparison (Blue: Reference, Red: Significant)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=8)
    
    # P値分布
    comparison_models = models[1:]
    comparison_pvals = p_values[1:]
    
    bars = ax2.bar(comparison_models, comparison_pvals, 
                   color=['red' if pv < 0.05 else 'gray' for pv in comparison_pvals], 
                   alpha=0.7)
    ax2.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
    ax2.set_ylabel('P-value')
    ax2.set_title('P-values vs Reference Model')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    for bar, pv in zip(bars, comparison_pvals):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{pv:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 有意性サマリー
    sig_counts = [sum(significant[1:]), len(significant) - 1 - sum(significant[1:])]
    labels = ['Significant', 'Non-significant']
    colors_pie = ['red', 'gray']
    
    ax3.pie(sig_counts, labels=labels, colors=colors_pie, autopct='%1.0f%%')
    ax3.set_title('Significance Summary')
    
    # AUC差分
    ref_auc = aucs[0]
    auc_diffs = [auc - ref_auc for auc in aucs[1:]]
    
    bars = ax4.bar(comparison_models, auc_diffs,
                   color=['red' if sig else 'gray' for sig in significant[1:]],
                   alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_ylabel('AUC Difference from Reference')
    ax4.set_title('AUC Differences')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    for bar, diff in zip(bars, auc_diffs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., 
                height + (0.001 if height >= 0 else -0.001),
                f'{diff:+.3f}', ha='center', 
                va='bottom' if height >= 0 else 'top', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_multi_results(all_results, results_summary, output_path):
    """複数比較結果をファイルに保存"""
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    detailed_results = convert_numpy(all_results)
    with open(output_path + '_detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(output_path + '_summary.csv', index=False)
    
    with open(output_path + '_multi_comparison_report.txt', 'w') as f:
        f.write("=== Multi-Model Hanley McNeil AUC Comparison Report ===\n\n")
        
        f.write("SUMMARY TABLE:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<25} {'AUC':<10} {'P-value':<10} {'Significant':<12} {'95% CI':<20}\n")
        f.write("-" * 80 + "\n")
        
        for result in results_summary:
            # None値を適切に処理
            p_value_str = f"{result['p_value']:.4f}" if result['p_value'] is not None else "N/A"
            ci_str = f"[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]" if result['ci_lower'] is not None else "N/A"
            
            f.write(f"{result['model_name']:<25} {result['auc']:<10.4f} "
                   f"{p_value_str:<10} {str(result['significant']):<12} {ci_str:<20}\n")
        
        f.write("-" * 80 + "\n\n")
        
        significant_count = sum([r['significant'] for r in results_summary[1:]])
        total_comparisons = len(results_summary) - 1
        
        f.write("STATISTICAL SIGNIFICANCE SUMMARY:\n")
        f.write(f"Total comparisons: {total_comparisons}\n")
        f.write(f"Significant differences: {significant_count}\n")
        f.write(f"Non-significant differences: {total_comparisons - significant_count}\n\n")
        
        f.write("DETAILED INTERPRETATIONS:\n\n")
        ref_name = results_summary[0]['model_name']
        ref_auc = results_summary[0]['auc']
        
        for i, result in enumerate(results_summary[1:], 1):  # 基準モデル(index=0)をスキップ
            f.write(f"{i}. {ref_name} vs {result['model_name']}:\n")
            f.write(f"   AUC Difference: {result['auc'] - ref_auc:+.4f}\n")
            f.write(f"   P-value: {result['p_value']:.6f}\n")
            
            if result['significant']:
                f.write("   ✓ SIGNIFICANT DIFFERENCE\n")
                if result['auc'] > ref_auc:
                    f.write(f"   → {result['model_name']} performs significantly better\n")
                else:
                    f.write(f"   → {ref_name} performs significantly better\n")
            else:
                f.write("   ✗ No significant difference\n")
            f.write("\n")

def extract_scores_for_csv(pred):
    """
    CSVファイル用のスコア抽出（既に正規化済みの確率用）
    predict.pyと同じ方法でAUCを計算
    """
    print(f"  Prediction array shape: {pred.shape}")
    print(f"  First few prediction values: {pred[:3]}")
    
    if pred.ndim > 1 and pred.shape[1] == 2:
        # 2列の場合：prob_class_0, prob_class_1（既に正規化済み）
        row_sums = np.sum(pred, axis=1)
        print(f"  Row sums (first 5): {row_sums[:5]}")
        
        # 確率が正規化済みかチェック（合計が約1.0）
        if np.allclose(row_sums, 1.0, atol=0.01):
            print("  ✓ Confirmed: These are normalized probabilities")
            print("  → Using prob_class_1 directly (same as predict.py)")
            # predict.pyと同じ：正規化済み確率のクラス1をそのまま使用
            return pred[:, 1]
        else:
            print("  ⚠️  Values don't sum to 1 - these might be logits")
            print("  → Applying softmax to convert to probabilities")
            # Logitsの場合はSoftmaxを適用
            softmax_probs = softmax(pred)
            return softmax_probs[:, 1]
    elif pred.ndim == 1:
        print("  Single column - using as-is")
        return pred
    else:
        print("  Multiple columns - applying softmax")
        return softmax(pred)[:, 1]

def run_multi_analysis():
    """複数モデルの一括比較分析を実行"""
    print("=== Multi-Model Hanley McNeil AUC Comparison Test ===\n")
    print("Using predefined paths from DEFAULT_PATHS configuration...")
    
    # 設定表示
    print("\n📁 Configuration:")
    print(f"  Reference: {DEFAULT_PATHS['reference']} ({DEFAULT_PATHS['reference_name']})")
    print(f"  Comparison models: {len(DEFAULT_PATHS['models'])}")
    for i, model in enumerate(DEFAULT_PATHS['models']):
        print(f"    {i+1}. {model['path']} ({model['name']})")
    print()
    
    # 基準データの読み込み
    print("Loading reference model...")
    try:
        ref_pred, ref_true, _ = load_prediction_results(DEFAULT_PATHS['reference'])
        print(f"✓ {DEFAULT_PATHS['reference_name']} loaded successfully")
    except Exception as e:
        print(f"✗ Error loading reference model: {e}")
        print(f"Please check the file format and content structure.")
        return
    
    # 基準データのスコア抽出（修正版）
    ref_scores = extract_scores_for_csv(ref_pred)
    ref_auc = roc_auc_score(ref_true, ref_scores)
    
    print(f"\n🎯 Reference Model Summary:")
    print(f"  {DEFAULT_PATHS['reference_name']}: AUC = {ref_auc:.6f}, n = {len(ref_true)}")
    print(f"  Positive: {np.sum(ref_true == 1)} ({100*np.sum(ref_true == 1)/len(ref_true):.1f}%)")
    print(f"  Score range: [{np.min(ref_scores):.6f}, {np.max(ref_scores):.6f}]")
    
    # predict.pyと同じAUC値になっているか検証用の情報
    print(f"\n📊 AUC Calculation Verification:")
    print(f"  Using scores (class 1 probabilities): {ref_scores[:5]} ...")
    print(f"  True labels: {ref_true[:5]} ...")
    print(f"  → This should match predict.py AUC value: {ref_auc:.6f}")
    
    # 比較モデルの処理
    all_results = []
    comparison_data = []
    results_summary = []
    
    # 基準モデルをサマリーに追加
    results_summary.append({
        'model_name': DEFAULT_PATHS['reference_name'],
        'auc': ref_auc,
        'p_value': None,
        'significant': False,
        'ci_lower': None,
        'ci_upper': None,
        'sample_size': len(ref_true)
    })
    
    print(f"\n🔬 Performing comparisons...")
    
    for i, model_config in enumerate(DEFAULT_PATHS['models']):
        model_path = model_config['path']
        model_name = model_config['name']
        
        print(f"\n--- Comparison {i+1}: {DEFAULT_PATHS['reference_name']} vs {model_name} ---")
        
        try:
            # データ読み込み
            comp_pred, comp_true, _ = load_prediction_results(model_path)
            comp_scores = extract_scores_for_csv(comp_pred)  # 修正版を使用
            comp_auc = roc_auc_score(comp_true, comp_scores)
            
            print(f"  {model_name}: AUC = {comp_auc:.6f}, n = {len(comp_true)}")
            print(f"  Score range: [{np.min(comp_scores):.6f}, {np.max(comp_scores):.6f}]")
            
            # Hanley McNeil検定実行
            results = hanley_mcneil_independent_test(
                ref_true, ref_scores, comp_true, comp_scores, DEFAULT_PATHS['alpha']
            )
            
            # ブートストラップ検定（オプション）
            bootstrap_results = {}
            if DEFAULT_PATHS['bootstrap']:
                print("    Running bootstrap test...")
                bootstrap_results = bootstrap_auc_comparison_independent(
                    ref_true, ref_scores, comp_true, comp_scores, alpha=DEFAULT_PATHS['alpha']
                )
            
            # 結果保存
            comparison_result = {
                'comparison_id': i + 1,
                'reference_name': DEFAULT_PATHS['reference_name'],
                'comparison_name': model_name,
                'hanley_mcneil': results,
                'bootstrap': bootstrap_results
            }
            all_results.append(comparison_result)
            
            # プロット用データ
            comparison_data.append((comp_true, comp_scores, model_name))
            
            # サマリー用データ
            results_summary.append({
                'model_name': model_name,
                'auc': comp_auc,
                'p_value': results['p_value'],
                'significant': results['significant'],
                'ci_lower': results['ci_lower'],
                'ci_upper': results['ci_upper'],
                'sample_size': len(comp_true)
            })
            
            # 結果表示
            print(f"    AUC difference: {comp_auc - ref_auc:+.6f}")
            print(f"    P-value: {results['p_value']:.6f}")
            print(f"    Significant: {'YES ✓' if results['significant'] else 'NO ✗'}")
            
        except Exception as e:
            print(f"  ✗ Error processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 結果サマリー表示
    print("\n" + "="*80)
    print("=== FINAL RESULTS SUMMARY ===")
    print("="*80)
    
    print(f"{'Model':<25} {'AUC':<12} {'P-value':<12} {'Significant':<12}")
    print("-" * 65)
    
    for result in results_summary:
        pval_str = f"{result['p_value']:.6f}" if result['p_value'] is not None else "N/A"
        print(f"{result['model_name']:<25} {result['auc']:<12.6f} "
              f"{pval_str:<12} {str(result['significant']):<12}")
    
    # 統計的サマリー
    significant_count = sum([r['significant'] for r in results_summary[1:]])
    total_comparisons = len(results_summary) - 1
    
    print(f"\n📊 Statistical Summary:")
    print(f"  Total comparisons: {total_comparisons}")
    print(f"  Significant differences: {significant_count}")
    print(f"  Non-significant differences: {total_comparisons - significant_count}")
    
    # 結果保存
    print(f"\n💾 Saving results to: {DEFAULT_PATHS['output']}")
    output_dir = Path(DEFAULT_PATHS['output'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_base = output_dir / "multi_comparison"
    save_multi_results(all_results, results_summary, str(output_base))
    
    # プロット作成
    if comparison_data:
        detailed_plot_path = output_base.with_name("detailed_comparisons.png")
        create_multi_comparison_plot(
            (ref_true, ref_scores, DEFAULT_PATHS['reference_name']),
            comparison_data,
            str(detailed_plot_path)
        )
        
        summary_plot_path = output_base.with_name("summary_plot.png")
        create_summary_plot(results_summary, str(summary_plot_path))
        
        print(f"\n📄 Output Files:")
        print(f"  Detailed results (JSON): {output_base}_detailed_results.json")
        print(f"  Summary (CSV): {output_base}_summary.csv")
        print(f"  Report (TXT): {output_base}_multi_comparison_report.txt")
        print(f"  Detailed plots: {detailed_plot_path}")
        print(f"  Summary plot: {summary_plot_path}")
    
    print(f"\n🎉 Multi-model analysis completed successfully!")

def main():
    """メイン関数"""
    run_multi_analysis()

if __name__ == '__main__':
    main()