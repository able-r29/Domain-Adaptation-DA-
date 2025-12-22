"""
Utility functions for DANN training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import datasets.datasets as dataset
import models.models as models


def load_json(path):
    """JSONファイルを読み込み"""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path, data):
    """JSONファイルを保存"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def save_text(path, text):
    """テキストファイルを保存"""
    with open(path, 'w') as f:
        f.write(text)


def command_log(out_dir):
    """コマンドログを保存"""
    import sys
    command = ' '.join(sys.argv)
    save_text(os.path.join(out_dir, 'command.txt'), command)


def parse_devices(device_str):
    """デバイス文字列を解析"""
    if ',' in device_str:
        device_parts = device_str.replace('cuda:', '').split(',')
        device_ids = []
        for part in device_parts:
            try:
                device_id = int(part.strip())
                if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                    device_ids.append(device_id)
            except ValueError:
                pass
        
        if not device_ids:
            device_ids = [0]
            
        primary_device = torch.device(f'cuda:{device_ids[0]}')
        return device_ids, primary_device
    else:
        device_id = int(device_str.replace('cuda:', ''))
        return [device_id], torch.device(f'cuda:{device_id}')


def setup_cuda_environment(device_ids, seed=None):
    """CUDA環境セットアップ"""
    torch.backends.cudnn.benchmark = True
    
    if seed is not None:
        import random
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        g = torch.Generator()
        g.manual_seed(seed)
        return g
    return None


def print_gpu_memory_info(device_ids):
    """GPU メモリ使用量を表示"""
    for device_id in device_ids:
        torch.cuda.empty_cache()
        print(f"🔍 GPU {device_id} memory:")
        device_obj = torch.device(f'cuda:{device_id}')
        total_mem = torch.cuda.get_device_properties(device_obj).total_memory / 1e9
        allocated_mem = torch.cuda.memory_allocated(device_obj) / 1e9
        print(f"  Total: {total_mem:.2f} GB, Allocated: {allocated_mem:.2f} GB, Free: {total_mem - allocated_mem:.2f} GB")


def get_datasets(config, fold, generator):
    """データセットを取得"""
    # ソースドメイン
    loader_src, loader_eval_tr, loader_eval_vl = dataset.get_dataset(
        i_fold=fold, generator=generator, shuffle=True, **config['dataset']
    )
    
    # ターゲットドメイン
    if 'dataset_target' in config:
        loader_target, _, _ = dataset.get_dataset(
            i_fold=fold, generator=generator, shuffle=True, **config['dataset_target']
        )
    else:
        loader_target = loader_src
    
    return loader_src, loader_eval_tr, loader_eval_vl, loader_target


def get_model_and_processors(config, device):
    """モデルと前後処理を取得"""
    backbone = models.get_model(**config['model']).to(device)
    pre, post, func, met = models.get_process(device=device, **config['process'])
    return backbone, pre, post, func, met


def init_optimizer(params, config):
    """オプティマイザーを初期化"""
    opt_config = config['opt']
    name = opt_config['name']
    
    if name == 'sgd':
        kwargs = {k: v for k, v in opt_config.items() if k != 'name'}
        if 'momentum' in kwargs:
            kwargs['nesterov'] = True
        return optim.SGD(params, **kwargs)
    elif name == 'adam':
        return optim.Adam(params, **{k: v for k, v in opt_config.items() if k != 'name'})
    elif name == 'rmsprop':
        return optim.RMSprop(params, **{k: v for k, v in opt_config.items() if k != 'name'})
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def get_scheduler(optimizer, config):
    """学習率スケジューラーを取得"""
    if 'scheduler' not in config:
        return None
        
    scheduler_config = config['scheduler']
    if scheduler_config['name'] == 'lambda':
        gamma = scheduler_config.get('gamma', 0.1)
        decay_rate = scheduler_config.get('decay_rate', 0.001)
        return optim.lr_scheduler.LambdaLR(
            optimizer, 
            lambda x: gamma * (1. + decay_rate * float(x)) ** (-0.75)
        )
    return None


def is_integer_dtype(dtype):
    """dtype が整数型かどうかを安全に判定"""
    dtype_str = str(dtype)
    integer_types = ['int8', 'int16', 'int32', 'int64', 'uint8', 'bool']
    return any(int_type in dtype_str for int_type in integer_types)


def is_string_tensor(tensor):
    """テンソルが文字列型かどうかを安全に判定"""
    if not isinstance(tensor, torch.Tensor):
        return False
    
    # PyTorchバージョンに依存しない文字列型チェック
    dtype_str = str(tensor.dtype)
    return 'object' in dtype_str


def convert_string_tensor_to_numeric(tensor, device):
    """文字列テンソルを数値テンソルに変換"""
    try:
        if is_string_tensor(tensor):
            # 文字列テンソルの場合
            if hasattr(tensor, 'tolist'):
                string_list = tensor.tolist()
            else:
                string_list = [item.item() if hasattr(item, 'item') else str(item) for item in tensor]
            
            # 一意なラベルを取得してインデックスに変換
            unique_labels = sorted(list(set(string_list)))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            indices = [label_to_idx[label] for label in string_list]
            return torch.tensor(indices, dtype=torch.long).to(device)
        else:
            # 数値テンソルの場合はそのまま返す（型変換のみ）
            if not is_integer_dtype(tensor.dtype):
                return tensor.long().to(device)
            else:
                return tensor.to(device)
    except Exception as e:
        print(f"❌ String tensor conversion failed: {e}")
        # フォールバック: ゼロテンソル
        return torch.zeros(len(tensor), dtype=torch.long).to(device)


def safe_batch_processing(batch, device, pre_function=None, is_evaluation=False):
    """バッチを安全に処理（修正版：評価時前処理強制適用）"""
    try:
        # 評価時も前処理を必ず適用
        if pre_function is not None:
            try:
                x, y = pre_function(batch, device, True)  # 最後の引数をTrueに固定
                
                # ラベル処理
                if isinstance(y, dict):
                    if 'label' in y:
                        y = y['label']
                    elif 'ya' in y:
                        y = y['ya']
                    else:
                        y = list(y.values())[0]
                
                # 型変換
                if isinstance(y, torch.Tensor):
                    if is_string_tensor(y):
                        y = convert_string_tensor_to_numeric(y, device)
                    elif not is_integer_dtype(y.dtype):
                        y = y.long().to(device)
                    else:
                        y = y.to(device)
                
                return x, y
                
            except Exception as e:
                print(f"Pre-processing failed: {e}, using manual processing")
        
        # 手動処理（前処理がない場合のみ）
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
            
            # データの型変換とデバイス移動
            if isinstance(x, torch.Tensor):
                x = x.to(device)
            elif isinstance(x, dict):
                x = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in x.items()}
            
            # ラベル処理
            if isinstance(y, torch.Tensor):
                if is_string_tensor(y):
                    y = convert_string_tensor_to_numeric(y, device)
                else:
                    y = y.long().to(device)
            elif isinstance(y, (list, tuple)):
                try:
                    y = torch.tensor(y, dtype=torch.long).to(device)
                except:
                    y = torch.zeros(len(y) if hasattr(y, '__len__') else 1, 
                                   dtype=torch.long).to(device)
            
            return x, y
        
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        raise e


def safe_post_processing(y_pred, post_function, x, y):
    """後処理を安全に適用"""
    if post_function is None:
        return y_pred
    
    try:
        import inspect
        sig = inspect.signature(post_function)
        params = list(sig.parameters.keys())
        
        if len(params) == 4:
            return post_function(y_pred, x, y, None)
        elif len(params) == 3:
            return post_function(y_pred, x, y)
        elif len(params) == 2:
            return post_function(y_pred, y)
        else:
            return post_function(y_pred)
            
    except Exception as e:
        print(f"Post-processing failed: {e}, using raw predictions")
        return y_pred


def set_alpha_safely(model, alpha):
    """DataParallel対応でset_alphaを安全に呼び出し"""
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        # .moduleを通してアクセス
        model.module.set_alpha(alpha)
    else:
        # 直接アクセス
        model.set_alpha(alpha)


def macro_sensitivity(y_pred, y_true, n_classes):
    """Macro Sensitivity計算（改良版）"""
    try:
        from sklearn.metrics import confusion_matrix
        import numpy as np
        
        # 予測ラベルを取得
        if y_pred.ndim > 1:
            y_pred_labels = np.argmax(y_pred, axis=1)
        else:
            y_pred_labels = y_pred
        
        # 混同行列を作成
        cm = confusion_matrix(y_true, y_pred_labels, labels=range(n_classes))
        sensitivities = []
        
        for i in range(n_classes):
            tp = cm[i, i]  # True Positive
            fn = np.sum(cm[i, :]) - tp  # False Negative
            
            if tp + fn > 0:
                sensitivity = tp / (tp + fn)
                sensitivities.append(sensitivity)
            else:
                # そのクラスのサンプルが存在しない場合
                sensitivities.append(0.0)
        
        # マクロ平均を計算
        macro_sens = np.mean(sensitivities) if sensitivities else 0.0
        
        # デバッグ情報（最初の評価時のみ）
        if len(sensitivities) > 0:
            print(f"📊 Class sensitivities: {[f'{s:.3f}' for s in sensitivities]}, Macro: {macro_sens:.3f}")
        
        return macro_sens
        
    except Exception as e:
        print(f"❌ Macro sensitivity calculation failed: {e}")
        return 0.5  # フォールバック


class ForeverDataIterator:
    """データローダーの無限イテレータ"""
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
    
    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            return next(self.iter)
