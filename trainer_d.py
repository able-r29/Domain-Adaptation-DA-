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

from domain_discriminator import DomainDiscriminator, DANNClassifier, calculate_lambda_p
import utils


def create_train_step(classifier, domain_discriminator, optimizer, scheduler, 
                     iter_target, device, config):
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
            alpha = 1.0  # 固定値
            utils.set_alpha_safely(domain_discriminator, alpha)
            
            # デバッグ情報でalpha値を確認
            if engine.state.epoch <= 2 and engine.state.iteration <= 3:
                print(f"  GRL alpha: {alpha:.4f} (fixed)")
            
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
                    from sklearn.metrics import roc_auc_score
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


def create_evaluation_step(classifier, domain_discriminator, device, config):
    """評価ステップ関数（ドメインAUC修正版）"""
    # 前処理関数を事前に取得
    _, pre, _, _, _ = utils.get_model_and_processors(config, device)
    
    # 損失関数を定義
    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()
    
    def evaluation_step(engine, batch):
        classifier.eval()
        domain_discriminator.eval()
        with torch.no_grad():
            try:
                x, y = utils.safe_batch_processing(batch, device, pre, is_evaluation=True)
                
                # ★ 疾患分類の評価（ソースデータのみ）
                y_pred, features = classifier(x)
                
                cls_loss = cls_criterion(y_pred, y)
                correct = (y_pred.argmax(dim=1) == y).float().mean()
                
                # CPUに移動してnumpy配列に変換
                y_pred_np = y_pred.detach().cpu().numpy()
                y_true_np = y.detach().cpu().numpy()
                
                # macro-sensitivity計算
                n_classes = y_pred.shape[1]
                macro_sens = utils.macro_sensitivity(y_pred_np, y_true_np, n_classes)
                
                # AUC計算
                try:
                    from sklearn.metrics import roc_auc_score
                    if n_classes == 2:
                        y_pred_prob = torch.softmax(y_pred, dim=1)[:, 1].detach().cpu().numpy()
                        auc = roc_auc_score(y_true_np, y_pred_prob)
                    else:
                        y_pred_prob = torch.softmax(y_pred, dim=1).detach().cpu().numpy()
                        auc = roc_auc_score(y_true_np, y_pred_prob, multi_class='ovr')
                except Exception as e:
                    auc = 0.5
                
                return {
                    "cls_loss": cls_loss.item(),
                    "cls_accuracy": correct.item(),
                    "cls_auc": auc,
                    "cls_macro_sensitivity": macro_sens,
                    "source_features": features,  # ★ ソース特徴を返す
                    "source_batch_size": x.shape[0] if isinstance(x, torch.Tensor) else next(iter(x.values())).shape[0]
                }
                    
            except Exception as e:
                print(f"❌ Evaluation step failed: {e}")
                return {
                    "cls_loss": 1.0, "cls_accuracy": 0.0, "cls_auc": 0.5, "cls_macro_sensitivity": 0.0,
                    "source_features": torch.zeros(1, 256).to(device), "source_batch_size": 1
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
    train_step = create_train_step(classifier, domain_discriminator, optimizer, scheduler,
                                  iter_target, primary_device, config)
    trainer = Engine(train_step)
    
    # 評価エンジン作成（修正版）
    evaluation_step = create_evaluation_step(classifier, domain_discriminator, primary_device, config)
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
        
        # 学習時のメトリクス（毎エポック）
        train_metrics = {
            "epoch": engine.state.epoch,
            
            # ★ 学習時 - 疾患分類
            "train/cls_loss": out['cls_loss'],
            "train/cls_accuracy": out['cls_acc'],
            "train/cls_auc": out['cls_auc'],
            "train/cls_macro_sensitivity": out['cls_macro_sensitivity'],
            
            # ★ 学習時 - ドメイン識別
            "train/domain_loss": out['domain_loss'],
            "train/domain_accuracy": out['domain_acc'],
            "train/domain_auc": out['domain_auc'],
            
            # ★ 学習時 - モデル全体損失
            "train/loss": out['loss']
        }
        
        print(f"Epoch {engine.state.epoch:3d} - "
              f"Loss: {out['loss']:.4f} "
              f"(Cls: {out['cls_loss']:.4f}, Domain: {out['domain_loss']:.4f}) | "
              f"Cls Acc: {out['cls_acc']:.3f}, Cls AUC: {out['cls_auc']:.3f}, "
              f"Domain Acc: {out['domain_acc']:.3f}, Domain AUC: {out['domain_auc']:.3f}")
        
        # 評価実行（毎エポック）
        classifier.eval()
        domain_discriminator.eval()
        try:
            # Train評価
            eval_tr.run(loader_eval_tr, max_epochs=1)
            train_eval_output = eval_tr.state.output
            
            # Validation評価
            eval_vl.run(loader_eval_vl, max_epochs=1)
            val_output = eval_vl.state.output
            
            # ★ ドメインAUC評価（ソース+ターゲット混合）
            train_domain_auc, train_domain_acc, train_domain_loss = domain_evaluation_step(
                train_eval_output["source_features"], train_eval_output["source_batch_size"]
            )
            
            val_domain_auc, val_domain_acc, val_domain_loss = domain_evaluation_step(
                val_output["source_features"], val_output["source_batch_size"]
            )
            
            # 評価時のメトリクスを追加
            train_metrics.update({
                # ★ Train評価 - 疾患分類
                "train_eval/cls_loss": train_eval_output["cls_loss"],
                "train_eval/cls_accuracy": train_eval_output["cls_accuracy"],
                "train_eval/cls_auc": train_eval_output["cls_auc"],
                "train_eval/cls_macro_sensitivity": train_eval_output["cls_macro_sensitivity"],
                
                # ★ Train評価 - ドメイン識別（修正版）
                "train_eval/domain_loss": train_domain_loss,
                "train_eval/domain_accuracy": train_domain_acc,
                "train_eval/domain_auc": train_domain_auc,
                
                # ★ Train評価 - モデル全体損失
                "train_eval/loss": train_eval_output["cls_loss"] + train_domain_loss,
                
                # ★ Validation評価 - 疾患分類
                "val/cls_loss": val_output["cls_loss"],
                "val/cls_accuracy": val_output["cls_accuracy"],
                "val/cls_auc": val_output["cls_auc"],
                "val/cls_macro_sensitivity": val_output["cls_macro_sensitivity"],
                
                # ★ Validation評価 - ドメイン識別（修正版）
                "val/domain_loss": val_domain_loss,
                "val/domain_accuracy": val_domain_acc,
                "val/domain_auc": val_domain_auc,
                
                # ★ Validation評価 - モデル全体損失
                "val/loss": val_output["cls_loss"] + val_domain_loss
            })
            
        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")
            # エラー時はデフォルト値を設定
            train_metrics.update({
                "train_eval/cls_loss": 1.0, "train_eval/cls_accuracy": 0.0, "train_eval/cls_auc": 0.5, "train_eval/cls_macro_sensitivity": 0.0,
                "train_eval/domain_loss": 1.0, "train_eval/domain_accuracy": 0.5, "train_eval/domain_auc": 0.5, "train_eval/loss": 2.0,
                "val/cls_loss": 1.0, "val/cls_accuracy": 0.0, "val/cls_auc": 0.5, "val/cls_macro_sensitivity": 0.0,
                "val/domain_loss": 1.0, "val/domain_accuracy": 0.5, "val/domain_auc": 0.5, "val/loss": 2.0
            })
        
        # wandbに記録
        wandb.log(train_metrics)
        
        classifier.train()
        domain_discriminator.train()
    
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