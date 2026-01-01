"""
DANN Training Script - Final Version
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import wandb
from ignite.engine import Events, Engine
from ignite.contrib.handlers import ProgressBar
from sklearn.metrics import roc_auc_score  # ←追加

from domain_discriminator import DomainDiscriminator, DANNClassifier, calculate_lambda_p
import utils
import numpy as np

def create_train_step(classifier, domain_discriminator, optimizer, scheduler, 
                    iter_target, device, config, loader_src):
    """学習ステップ関数（疾患分類メトリクス追加版）"""
    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()
    trade_off = config.get('dann', {}).get('trade_off', 1.0)
    max_epochs = config['train']['epoch']
    
    # 前処理関数を事前に取得
    _, pre, _, _, _ = utils.get_model_and_processors(config, device)
    
    def train_step(engine, batch):
        classifier.train()
        domain_discriminator.train()
        optimizer.zero_grad()
        
        try:
            # ソースとターゲットのバッチを取得
            x_s, y_s = utils.safe_batch_processing(batch, device, pre, is_evaluation=False)
            target_batch = next(iter_target)
            x_t, _ = utils.safe_batch_processing(target_batch, device, pre, is_evaluation=False)
            
            # 設定ファイルのバッチサイズを取得
            config_batch_size = config.get('dataset', {}).get('batch_size', 128)
            half_size = config_batch_size // 2  # 64
            
            # デバッグ情報
            if engine.state.epoch <= 2 and engine.state.iteration <= 3:
                print(f"🔍 Batch size debug:")
                print(f"  Config batch_size: {config_batch_size}")
                print(f"  Target half_size: {half_size}")
                if isinstance(x_s, torch.Tensor):
                    print(f"  Source actual size: {x_s.shape[0]}")
                elif isinstance(x_s, dict):
                    sample_tensor = next(iter(x_s.values()))
                    print(f"  Source actual size: {sample_tensor.shape[0]}")
                if isinstance(x_t, torch.Tensor):
                    print(f"  Target actual size: {x_t.shape[0]}")
                elif isinstance(x_t, dict):
                    sample_tensor = next(iter(x_t.values()))
                    print(f"  Target actual size: {sample_tensor.shape[0]}")
            
            # データの実際のサイズをチェック
            if isinstance(x_s, torch.Tensor):
                actual_source_size = x_s.shape[0]
            elif isinstance(x_s, dict):
                actual_source_size = next(iter(x_s.values())).shape[0]
            else:
                actual_source_size = len(x_s)
            
            if isinstance(x_t, torch.Tensor):
                actual_target_size = x_t.shape[0]
            elif isinstance(x_t, dict):
                actual_target_size = next(iter(x_t.values())).shape[0]
            else:
                actual_target_size = len(x_t)
            
            # ★ 十分なデータがない場合の警告
            if actual_source_size < half_size:
                print(f"⚠️ Warning: Source batch too small ({actual_source_size} < {half_size})")
                half_size = min(half_size, actual_source_size)
            
            if actual_target_size < half_size:
                print(f"⚠️ Warning: Target batch too small ({actual_target_size} < {half_size})")
                half_size = min(half_size, actual_target_size)
            
            if half_size < 8:  # 最小バッチサイズ確保
                print(f"❌ CRITICAL: Batch size too small: {half_size}")
                print(f"Source: {actual_source_size}, Target: {actual_target_size}")
                raise ValueError(f"Insufficient data for mixed batch: source={actual_source_size}, target={actual_target_size}")
            
            # 混合バッチを作成（ソース64 + ターゲット64 = 128）
            if isinstance(x_s, torch.Tensor) and isinstance(x_t, torch.Tensor):
                # テンソルの場合
                mixed_x = torch.cat([x_s[:half_size], x_t[:half_size]], dim=0)
                mixed_y_source = y_s[:half_size]
                
            elif isinstance(x_s, dict) and isinstance(x_t, dict):
                # 辞書の場合
                mixed_x = {}
                for key in x_s.keys():
                    if key in x_t and hasattr(x_s[key], 'shape') and hasattr(x_t[key], 'shape'):
                        mixed_x[key] = torch.cat([x_s[key][:half_size], x_t[key][:half_size]], dim=0)
                    else:
                        # キーが一致しない場合はソースデータのみ使用（警告出力）
                        print(f"⚠️ Key '{key}' not found in target, using source data only")
                        mixed_x[key] = x_s[key][:half_size]
                
                mixed_y_source = y_s[:half_size]
            else:
                raise ValueError(f"Incompatible data types: {type(x_s)} and {type(x_t)}")
            
            # ドメインラベルを作成（0: ソース[0:half_size], 1: ターゲット[half_size:2*half_size]）
            domain_labels = torch.cat([
                torch.zeros(half_size, 1),  # ソース部分
                torch.ones(half_size, 1)    # ターゲット部分
            ], dim=0).to(device)
            
            # デバッグ情報（成功時）
            if engine.state.epoch <= 2 and engine.state.iteration <= 3:
                print(f"✓ Mixed batch created successfully:")
                print(f"  Source samples: {half_size} (indices 0:{half_size})")
                print(f"  Target samples: {half_size} (indices {half_size}:{half_size*2})")
                print(f"  Total batch size: {half_size*2}")
                if isinstance(mixed_x, dict):
                    for k, v in mixed_x.items():
                        if hasattr(v, 'shape'):
                            print(f"  {k} shape: {v.shape}")
                else:
                    print(f"  Mixed batch shape: {mixed_x.shape}")
                print(f"  Domain labels shape: {domain_labels.shape}")
            
            # GRL強度調整
            p = float(engine.state.iteration + (engine.state.epoch - 1) * len(loader_src)) / (max_epochs * len(loader_src))
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0  # DANNの標準式
            
            # alpha = 1.0  # ★ この行を削除
            utils.set_alpha_safely(domain_discriminator, alpha)
            
            # デバッグ情報
            if engine.state.epoch <= 2 and engine.state.iteration <= 3:
                print(f"  GRL alpha: {alpha:.4f} (dynamic, p={p:.4f})")
            
            # 分類器で特徴抽出と分類
            mixed_pred, mixed_features = classifier(mixed_x)
            
            # 分類損失（ソース部分のみ：インデックス0からhalf_size-1）
            source_pred = mixed_pred[:half_size]
            cls_loss = cls_criterion(source_pred, mixed_y_source)
            
            # ★ ドメイン識別（GRLを通して勾配反転）
            # domain_discriminatorの内部でmixed_featuresがGRLを通過して勾配反転
            domain_pred = domain_discriminator(mixed_features)
            domain_loss = domain_criterion(domain_pred, domain_labels)
            
            # 総損失（重要：domain_lossはGRLにより分類器への勾配を反転）
            total_loss = cls_loss + trade_off * domain_loss
            
            # バックプロパゲーション
            # ここでGRLの勾配反転が効果を発揮：
            # - cls_lossの勾配：分類器を「良い分類」方向に更新
            # - domain_lossの勾配：GRLにより分類器を「ドメイン識別困難」方向に更新
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # メトリクス計算
            with torch.no_grad():
                # ★ 疾患分類メトリクス（学習時）
                source_pred_np = source_pred.detach().cpu().numpy()
                mixed_y_source_np = mixed_y_source.detach().cpu().numpy()
                
                # 分類精度
                source_acc = (source_pred.argmax(dim=1) == mixed_y_source).float().mean()
                
                # 分類AUC
                try:
                    n_classes = source_pred.shape[1]
                    if n_classes == 2:
                        source_pred_prob = torch.softmax(source_pred, dim=1)[:, 1].detach().cpu().numpy()
                        cls_auc = roc_auc_score(mixed_y_source_np, source_pred_prob)
                    else:
                        source_pred_prob = torch.softmax(source_pred, dim=1).detach().cpu().numpy()
                        cls_auc = roc_auc_score(mixed_y_source_np, source_pred_prob, multi_class='ovr')
                except Exception as e:
                    print(f"⚠️ Classification AUC calculation failed: {e}")
                    cls_auc = 0.5
                
                # 分類Macro-Sensitivity
                cls_macro_sens = utils.macro_sensitivity(source_pred_np, mixed_y_source_np, source_pred.shape[1])
                
                # ★ ドメイン識別メトリクス（学習時）
                domain_pred_binary = (domain_pred > 0.5).float()
                domain_acc = (domain_pred_binary == domain_labels).float().mean()
                
                # ドメイン識別AUC
                try:
                    domain_pred_np = domain_pred.detach().cpu().numpy().flatten()
                    domain_labels_np = domain_labels.detach().cpu().numpy().flatten()
                    domain_auc = roc_auc_score(domain_labels_np, domain_pred_np)
                except Exception as e:
                    print(f"⚠️ Domain AUC calculation failed: {e}")
                    domain_auc = 0.5
            
            return {
                # 全体損失
                "loss": total_loss.item(),
                
                # 疾患分類メトリクス（学習時）
                "cls_loss": cls_loss.item(),
                "cls_acc": source_acc.item(),
                "cls_auc": cls_auc,
                "cls_macro_sensitivity": cls_macro_sens,
                
                # ドメイン識別メトリクス（学習時）
                "domain_loss": domain_loss.item(),
                "domain_acc": domain_acc.item(),
                "domain_auc": domain_auc,
                
                # デバッグ情報
                "alpha": alpha,
                "batch_size": half_size * 2,
                "source_samples": half_size,
                "target_samples": half_size
            }
            
        except Exception as e:
            print(f"❌ Training step failed: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    return train_step


def create_evaluation_step(classifier, domain_discriminator, iter_target, device, config):
    """シンプルな評価ステップ（ドメイン損失含む）"""
    _, pre, _, _, _ = utils.get_model_and_processors(config, device)
    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()
    trade_off = config.get('dann', {}).get('trade_off', 1.0)
    
    def evaluation_step(engine, batch):
        classifier.eval()
        domain_discriminator.eval()
        
        with torch.no_grad():
            try:
                # ソースバッチ処理
                x_s, y_s = utils.safe_batch_processing(batch, device, pre, is_evaluation=True)
                
                # ターゲットバッチ取得
                target_batch = next(iter_target)
                x_t, _ = utils.safe_batch_processing(target_batch, device, pre, is_evaluation=True)
                
                # バッチサイズ調整
                batch_size = min(
                    x_s.shape[0] if isinstance(x_s, torch.Tensor) else next(iter(x_s.values())).shape[0],
                    x_t.shape[0] if isinstance(x_t, torch.Tensor) else next(iter(x_t.values())).shape[0]
                )
                
                # ソースの一部とターゲットの一部を混合
                if isinstance(x_s, dict) and isinstance(x_t, dict):
                    mixed_x = {}
                    for key in x_s.keys():
                        if key in x_t:
                            mixed_x[key] = torch.cat([
                                x_s[key][:batch_size//2],
                                x_t[key][:batch_size//2]
                            ], dim=0)
                        else:
                            mixed_x[key] = x_s[key][:batch_size//2]
                elif isinstance(x_s, torch.Tensor) and isinstance(x_t, torch.Tensor):
                    mixed_x = torch.cat([x_s[:batch_size//2], x_t[:batch_size//2]], dim=0)
                else:
                    # 型が一致しない場合はソースのみ
                    mixed_x = x_s
                    batch_size = x_s.shape[0] if isinstance(x_s, torch.Tensor) else next(iter(x_s.values())).shape[0]
                
                # 分類器で予測
                if isinstance(mixed_x, dict) and len(mixed_x) > len(x_s) if isinstance(x_s, dict) else True:
                    # 混合バッチの場合
                    mixed_pred, mixed_features = classifier(mixed_x)
                    source_pred = mixed_pred[:batch_size//2]
                    source_y = y_s[:batch_size//2]
                    
                    # ドメインラベル作成
                    domain_labels = torch.cat([
                        torch.zeros(batch_size//2, 1),  # ソース
                        torch.ones(batch_size//2, 1)    # ターゲット
                    ], dim=0).to(device)
                    
                    # ドメイン損失計算
                    domain_pred = domain_discriminator(mixed_features)
                    domain_loss = domain_criterion(domain_pred, domain_labels)
                else:
                    # ソースのみの場合
                    source_pred, mixed_features = classifier(x_s)
                    source_y = y_s
                    domain_loss = torch.tensor(0.0).to(device)
                
                # 分類損失計算
                cls_loss = cls_criterion(source_pred, source_y)
                total_loss = cls_loss + trade_off * domain_loss
                
                # メトリクス計算
                cls_acc = (source_pred.argmax(dim=1) == source_y).float().mean()
                
                # AUC計算
                try:
                    n_classes = source_pred.shape[1]
                    source_pred_prob = torch.softmax(source_pred, dim=1)
                    
                    if n_classes == 2:
                        auc = roc_auc_score(source_y.cpu().numpy(), source_pred_prob[:, 1].cpu().numpy())
                    else:
                        auc = roc_auc_score(source_y.cpu().numpy(), source_pred_prob.cpu().numpy(), multi_class='ovr')
                except Exception as e:
                    print(f"⚠️ AUC calculation failed: {e}")  # エラーメッセージも改善
                    auc = 0.5
                
                return {
                    "cls_loss": cls_loss.item(),
                    "domain_loss": domain_loss.item(),
                    "total_loss": total_loss.item(),
                    "cls_accuracy": cls_acc.item(),
                    "cls_auc": auc,
                    "cls_macro_sensitivity": utils.macro_sensitivity(
                        source_pred.cpu().numpy(), source_y.cpu().numpy(), source_pred.shape[1]
                    )
                }
                
            except Exception as e:
                print(f"❌ Evaluation step failed: {e}")
                return {
                    "cls_loss": 1.0, "domain_loss": 1.0, "total_loss": 2.0,
                    "cls_accuracy": 0.0, "cls_auc": 0.5, "cls_macro_sensitivity": 0.0
                }
    
    return evaluation_step


def create_domain_evaluation_step(domain_discriminator, iter_target, device, config):
    """ドメイン識別専用評価ステップ"""
    # 前処理関数を事前に取得
    _, pre, _, _, _ = utils.get_model_and_processors(config, device)
    domain_criterion = nn.BCELoss()
    
    def domain_evaluation_step(source_features, source_batch_size):
        """ソース特徴とターゲット特徴でドメインAUC評価"""
        domain_discriminator.eval()
        with torch.no_grad():
            try:
                # ターゲットバッチを取得
                target_batch = next(iter_target)
                x_t, _ = utils.safe_batch_processing(target_batch, device, pre, is_evaluation=True)
                
                # ターゲット特徴を抽出（分類器はevalモードのまま）
                if hasattr(domain_discriminator, 'module'):
                    # DataParallel の場合、分類器を取得
                    classifier = domain_discriminator.module.classifier if hasattr(domain_discriminator.module, 'classifier') else None
                else:
                    classifier = domain_discriminator.classifier if hasattr(domain_discriminator, 'classifier') else None
                
                if classifier is None:
                    # 分類器への参照がない場合、グローバルから取得（トリッキー）
                    print("⚠️ Warning: Cannot access classifier from domain_discriminator")
                    return 0.5, 0.5, 1.0  # デフォルト値
                
                _, target_features = classifier(x_t)
                
                # バッチサイズを統一
                min_size = min(source_batch_size, target_features.shape[0])
                source_features_eval = source_features[:min_size]
                target_features_eval = target_features[:min_size]
                
                # 混合特徴とドメインラベル
                mixed_features = torch.cat([source_features_eval, target_features_eval], dim=0)
                domain_labels = torch.cat([
                    torch.zeros(min_size, 1),  # ソース
                    torch.ones(min_size, 1)    # ターゲット
                ], dim=0).to(device)
                
                # ドメイン識別
                domain_pred = domain_discriminator(mixed_features)
                domain_acc = ((domain_pred > 0.5).float() == domain_labels).float().mean()
                domain_loss = domain_criterion(domain_pred, domain_labels)
                
                # ドメイン識別AUC
                try:
                    domain_pred_np = domain_pred.detach().cpu().numpy().flatten()
                    domain_labels_np = domain_labels.detach().cpu().numpy().flatten()
                    domain_auc = roc_auc_score(domain_labels_np, domain_pred_np)
                except Exception as e:
                    print(f"⚠️ Domain AUC calculation failed: {e}")
                    domain_auc = 0.5
                
                return domain_auc, domain_acc.item(), domain_loss.item()
                
            except Exception as e:
                print(f"❌ Domain evaluation failed: {e}")
                return 0.5, 0.5, 1.0
    
    return domain_evaluation_step


def main(fold, device_ids, primary_device, out_dir, parallel_mode, **config):
    """メイン学習関数"""
    
    print(f"🚀 Starting DANN Training")
    print(f"Device IDs: {device_ids}, Primary: {primary_device}")
    
    # 初期化
    wandb.init(
        project="ResNet18_DANN_final",
        name=f"dann_fold{fold}" if fold is not None else "dann_holdout",
        config=config,
        dir=out_dir,
        tags=["dann", "final"]
    )
    
    # CUDA環境セットアップ
    torch.cuda.set_device(primary_device)
    g = utils.setup_cuda_environment(device_ids, config['train']['seed'])
    
    # データセット読み込み
    print(f"📂 Loading datasets...")
    loader_src, loader_eval_tr, loader_eval_vl, loader_target = utils.get_datasets(config, fold, g)
    iter_target = utils.ForeverDataIterator(loader_target)
    
    # モデル構築
    print(f"🏗️ Building models...")
    backbone, pre, post, func, met = utils.get_model_and_processors(config, primary_device)
    
    # サンプルバッチでバックボーンの確認（デバッグ）
    print(f"🔍 Checking backbone compatibility...")
    sample_batch = next(iter(loader_src))
    x_sample, y_sample = utils.safe_batch_processing(sample_batch, primary_device, pre, is_evaluation=False)
    
    print(f"Sample input type: {type(x_sample)}")
    if isinstance(x_sample, dict):
        print(f"Sample input keys: {list(x_sample.keys())}")
        for k, v in x_sample.items():
            if hasattr(v, 'shape'):
                print(f"  {k} shape: {v.shape}")
    else:
        print(f"Sample input shape: {x_sample.shape}")
    
    # DANN設定
    num_classes = config['model']['n_class']
    bottleneck_dim = config.get('dann', {}).get('bottleneck_dim', 256)
    domain_hidden = config.get('dann', {}).get('domain_hidden_size', 1024)
    
    print(f"📊 DANN Configuration:")
    print(f"  Num classes: {num_classes}")
    print(f"  Bottleneck dim: {bottleneck_dim}")
    print(f"  Domain hidden: {domain_hidden}")
    
    # モデル作成
    classifier = DANNClassifier(backbone, num_classes, bottleneck_dim).to(primary_device)
    domain_discriminator = DomainDiscriminator(
        initial_feature_dim=bottleneck_dim,
        hidden_dim=domain_hidden
    ).to(primary_device)
    
    # GPU並列化
    if len(device_ids) > 1 and parallel_mode == 'DataParallel':
        classifier = DataParallel(classifier, device_ids=device_ids)
        domain_discriminator = DataParallel(domain_discriminator, device_ids=device_ids)
        print(f"✓ Using DataParallel on GPUs: {device_ids}")
    
    # 最適化設定
    all_params = list(classifier.parameters()) + list(domain_discriminator.parameters())
    optimizer = utils.init_optimizer(all_params, config)
    scheduler = utils.get_scheduler(optimizer, config)
    
    # エンジン作成
    train_step = create_train_step(classifier, domain_discriminator, optimizer, scheduler,iter_target, primary_device, config, loader_src)
    trainer = Engine(train_step)
    
    # 評価エンジン作成（修正版）
    evaluation_step = create_evaluation_step(classifier, domain_discriminator, iter_target, primary_device, config)
    domain_evaluation_step = create_domain_evaluation_step(domain_discriminator, iter_target, primary_device, config)
    eval_tr = Engine(evaluation_step)
    eval_vl = Engine(evaluation_step)
    
    # プログレスバー
    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {'loss': x['loss']})
    
    # ログ処理（修正版：正しいドメインAUC計算）
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        out = engine.state.output
        
        print(f"Epoch {engine.state.epoch:3d} - "
              f"Loss: {out['loss']:.4f} "
              f"(Cls: {out['cls_loss']:.4f}, Domain: {out['domain_loss']:.4f}) | "
              f"Alpha: {out['alpha']:.4f}")
        
        # 評価実行
        try:
            classifier.eval()
            domain_discriminator.eval()
            
            # Train評価
            eval_tr.run(loader_eval_tr, max_epochs=1)
            train_eval = eval_tr.state.output
            
            # Validation評価
            eval_vl.run(loader_eval_vl, max_epochs=1)
            val_eval = eval_vl.state.output
            
            # 学習状況をプリント
            print(f"  Train Eval - Loss: {train_eval['total_loss']:.4f}, "
                  f"Acc: {train_eval['cls_accuracy']:.3f}, AUC: {train_eval['cls_auc']:.3f}")
            print(f"  Val Eval   - Loss: {val_eval['total_loss']:.4f}, "
                  f"Acc: {val_eval['cls_accuracy']:.3f}, AUC: {val_eval['cls_auc']:.3f}")
            
            # wandbログ
            wandb.log({
                "epoch": engine.state.epoch,
                # 訓練時
                "train/loss": out['loss'],
                "train/cls_loss": out['cls_loss'],
                "train/domain_loss": out['domain_loss'],
                "train/cls_acc": out['cls_acc'],
                "train/alpha": out['alpha'],
                
                # 評価時
                "train_eval/total_loss": train_eval['total_loss'],
                "train_eval/cls_acc": train_eval['cls_accuracy'],
                "train_eval/cls_auc": train_eval['cls_auc'],
                
                "val/total_loss": val_eval['total_loss'],
                "val/cls_acc": val_eval['cls_accuracy'],
                "val/cls_auc": val_eval['cls_auc'],
            })
            
            classifier.train()
            domain_discriminator.train()
            
        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")
    
    # 学習実行
    try:
        max_epochs = config['train']['epoch']
        print(f"🚀 Starting training for {max_epochs} epochs")
        trainer.run(loader_src, max_epochs=max_epochs)
        print("✓ Training completed!")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, type=str)
    parser.add_argument('--fold', '-f', type=int, default=None)
    parser.add_argument('--device', '-d', required=True, type=str)
    parser.add_argument('--parallel', '-p', choices=['DataParallel', 'single'], 
                       default='DataParallel')
    args = parser.parse_args()
    
    # デバイス設定
    device_ids, primary_device = utils.parse_devices(args.device)
    
    # 出力ディレクトリ
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    args.out_dir = f'../logs/{config_name}_final_gpu{"_".join(map(str, device_ids))}'
    if args.fold is not None:
        args.out_dir = f"{args.out_dir}_fold{args.fold}"
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # 設定読み込み
    config = utils.load_json(args.config)
    utils.save_json(os.path.join(args.out_dir, 'config.json'), config)
    utils.command_log(args.out_dir)
    
    # 実行
    main(args.fold, device_ids, primary_device, args.out_dir, args.parallel, **config)
    utils.save_text(os.path.join(args.out_dir, 'finish.txt'), '')