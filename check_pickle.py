# check_results.py
import pickle
import numpy as np

# Pickleファイルを開く
with open('../resnet18_0.1_es_final/predict.pickle', 'rb') as f:
    predictions, true_labels, file_paths = pickle.load(f)

# 結果を表示
print(f"データ件数: {len(file_paths)}")
print(f"予測値の形状: {predictions.shape}")
print(f"真のラベル: {true_labels.shape}")

# 最初の5件を表示
for i in range(min(5, len(file_paths))):
    print(f"\nファイル: {file_paths[i]}")
    print(f"真のラベル: {true_labels[i]}")
    print(f"予測値: {predictions[i]}")
    print(f"予測クラス: {np.argmax(predictions[i])}")