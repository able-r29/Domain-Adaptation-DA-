"""
DANN Training Script - resnet18_base対応版
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import wandb
from ignite.engine import Events, Engine
from ignite.contrib.handlers import ProgressBar
from sklearn.metrics import roc_auc_score

from domain_discriminator import DomainDiscriminator, DANNClassifier, calculate_lambda_p
import utils
import numpy as np


def create_train_step(classifier, domain_discriminator, optimizer, scheduler, 
                     iter_target, device, config, loader_src):
    """DANN学習ステップ（resnet18_base対応）"""
    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()
    max_epochs = config['train']['epoch']
    
    # 前処理関数を取得
    _, pre, _, _, _ = utils.get_model_and_processors(config, device)
    
    def train_step(engine, batch):
        classifier.train()
        domain_discriminator.train()
        optimizer.zero_grad()
        
        try:
            # ソース・ターゲットデータ取得
            x_s, y_s = utils.safe_batch_processing(batch, device, pre, is_evaluation=False)
            target_batch = next(iter_target)
            x_t, _ = utils.safe_batch_processing(target_batch, device, pre, is_evaluation=False)
            
            # バッチサイズ調整
            batch_size = min(x_s.shape[0], x_t.shape[0])
            half_size = batch_size // 2
            
            if half_size < 8:
                raise ValueError(f"Batch size too small: {batch_size}")
            
            # 混合バッチ作成（resnet18_baseはテンソル入力）
            mixed_x = torch.cat([x_s[:half_size], x_t[:half_size]], dim=0)
            mixed_y_source = y_s[:half_size]
            
            # ドメインラベル（0: ソース, 1: ターゲット）
            domain_labels = torch.cat([
                torch.zeros(half_size, 1),
                torch.ones(half_size, 1)
            ], dim=0).to(device)
            
            # GRL強度を動的調整（DANN論文に従って）
            p = float(engine.state.iteration + (engine.state.epoch - 1) * len(loader_src)) / (max_epochs * len(loader_src))
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            
            # alpha値をドメイン識別器に設定
            utils.set_alpha_safely(domain_discriminator, alpha)
            
            # デバッグ情報（初期エポックのみ）
            if engine.state.epoch <= 2 and engine.state.iteration <= 3:
                print(f"🔍 Training step debug:")
                print(f"  Mixed batch shape: {mixed_x.shape}")
                print(f"  Domain labels shape: {domain_labels.shape}")
                print(f"  GRL alpha: {alpha:.4f} (p={p:.4f})")
            
            # 分類器でforward（特徴抽出+分類）
            mixed_pred, mixed_features = classifier(mixed_x)
            
            # 分類損失（ソース部分のみ）
            source_pred = mixed_pred[:half_size]
            cls_loss = cls_criterion(source_pred, mixed_y_source)
            
            # ドメイン識別（mixed_featuresがGRLを通って勾配反転）
            domain_pred = domain_discriminator(mixed_features)
            domain_loss = domain_criterion(domain_pred, domain_labels)
            
            # 総損失
            total_loss = cls_loss +  domain_loss
            
            # バックプロパゲーション
            total_loss.backward()
            
            # 勾配クリッピング（安定化）
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(domain_discriminator.parameters(), 1.0)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # メトリクス計算
            with torch.no_grad():
                # 分類精度
                source_acc = (source_pred.argmax(dim=1) == mixed_y_source).float().mean()
                
                # 分類AUC
                try:
                    if source_pred.shape[1] == 2:
                        source_pred_prob = torch.softmax(source_pred, dim=1)[:, 1].cpu().numpy()
                        cls_auc = roc_auc_score(mixed_y_source.cpu().numpy(), source_pred_prob)
                    else:
                        source_pred_prob = torch.softmax(source_pred, dim=1).cpu().numpy()
                        cls_auc = roc_auc_score(mixed_y_source.cpu().numpy(), source_pred_prob, multi_class='ovr')
                except:
                    cls_auc = 0.5
                
                # ドメイン識別精度
                domain_acc = ((domain_pred > 0.5).float() == domain_labels).float().mean()
                
                # ドメイン識別AUC
                try:
                    domain_auc = roc_auc_score(
                        domain_labels.cpu().numpy().flatten(),
                        domain_pred.cpu().numpy().flatten()
                    )
                except:
                    domain_auc = 0.5
            
            return {
                "loss": total_loss.item(),
                "cls_loss": cls_loss.item(),
                "domain_loss": domain_loss.item(),
                "cls_acc": source_acc.item(),
                "cls_auc": cls_auc,
                "domain_acc": domain_acc.item(),
                "domain_auc": domain_auc,
                "alpha": alpha,
            }
            
        except Exception as e:
            print(f"❌ Training step failed: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    return train_step


def create_evaluation_step(classifier, domain_discriminator, loader_target, device, config):
    """修正版：評価ステップ（GRL適用制御付き）"""
    _, pre, _, _, _ = utils.get_model_and_processors(config, device)
    cls_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()
    
    def evaluation_step(engine, batch):
        classifier.eval()
        domain_discriminator.eval()
        
        with torch.no_grad():
            try:
                # ソースデータ処理
                x_s, y_s = utils.safe_batch_processing(batch, device, pre, is_evaluation=True)
                source_pred, source_features = classifier(x_s)
                cls_loss = cls_criterion(source_pred, y_s)
                source_cls_acc = (source_pred.argmax(dim=1) == y_s).float().mean()
                
                # AUC計算
                try:
                    if source_pred.shape[1] == 2:
                        source_pred_prob = torch.softmax(source_pred, dim=1)[:, 1].cpu().numpy()
                        source_auc = roc_auc_score(y_s.cpu().numpy(), source_pred_prob)
                    else:
                        source_pred_prob = torch.softmax(source_pred, dim=1).cpu().numpy()
                        source_auc = roc_auc_score(y_s.cpu().numpy(), source_pred_prob, multi_class='ovr')
                except:
                    source_auc = 0.5
                
                # ターゲットデータ取得
                try:
                    target_batch = next(loader_target)
                    x_t, _ = utils.safe_batch_processing(target_batch, device, pre, is_evaluation=True)
                    
                    min_batch_size = min(x_s.shape[0], x_t.shape[0])
                    x_s_eval = x_s[:min_batch_size]
                    x_t_eval = x_t[:min_batch_size]
                    y_s_eval = y_s[:min_batch_size]
                    
                    # ターゲット特徴抽出
                    target_pred, target_features = classifier(x_t_eval)
                    
                    # ★ 重要：評価時も訓練時と同じ条件でドメイン評価
                    mixed_features = torch.cat([source_features[:min_batch_size], target_features], dim=0)
                    domain_labels = torch.cat([
                        torch.zeros(min_batch_size, 1),
                        torch.ones(min_batch_size, 1)
                    ], dim=0).to(device)
                    
                    # ★ 評価時もGRLの現在のalpha値を使用
                    # alphaを現在の訓練進行度に基づいて計算
                    if hasattr(engine, 'state') and hasattr(engine.state, 'epoch'):
                        current_epoch = engine.state.epoch
                        max_epochs = config['train']['epoch']
                        loader_size = 6  # loader_eval_trのサイズ
                        
                        # 訓練と同じalpha計算
                        p = float(current_epoch) / max_epochs
                        eval_alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
                        
                        # ドメイン識別器にalpha設定
                        utils.set_alpha_safely(domain_discriminator, eval_alpha)
                        
                        print(f"  📊 Evaluation alpha: {eval_alpha:.4f} (epoch {current_epoch})")
                    
                    # ドメイン識別予測（GRL適用）
                    domain_pred = domain_discriminator(mixed_features)
                    domain_loss = domain_criterion(domain_pred, domain_labels)
                    
                    # ドメイン性能計算
                    domain_acc = ((domain_pred > 0.5).float() == domain_labels).float().mean()
                    
                    try:
                        domain_auc = roc_auc_score(
                            domain_labels.cpu().numpy().flatten(),
                            domain_pred.cpu().numpy().flatten()
                        )
                    except:
                        domain_auc = 0.5
                        
                except Exception as target_error:
                    print(f"⚠️ Target evaluation failed: {target_error}")
                    domain_labels = torch.zeros(x_s.shape[0], 1).to(device)
                    domain_pred = domain_discriminator(source_features)
                    domain_loss = domain_criterion(domain_pred, domain_labels)
                    domain_acc = ((domain_pred > 0.5).float() == domain_labels).float().mean()
                    domain_auc = 0.5
                
                # 総損失
                total_loss = cls_loss + domain_loss
                
                # Macro sensitivity
                try:
                    cls_macro_sensitivity = utils.macro_sensitivity(
                        source_pred.cpu().numpy(), y_s.cpu().numpy(), source_pred.shape[1]
                    )
                except:
                    cls_macro_sensitivity = 0.0
                
                return {
                    "cls_loss": cls_loss.item(),
                    "cls_accuracy": source_cls_acc.item(),
                    "cls_auc": source_auc,
                    "cls_macro_sensitivity": cls_macro_sensitivity,
                    "domain_loss": domain_loss.item(),
                    "domain_accuracy": domain_acc.item(),
                    "domain_auc": domain_auc,
                    "total_loss": total_loss.item(),
                }
                
            except Exception as e:
                print(f"❌ Evaluation step failed: {e}")
                return {
                    "cls_loss": 1.0,
                    "cls_accuracy": 0.0,
                    "cls_auc": 0.5,
                    "cls_macro_sensitivity": 0.0,
                    "domain_loss": 1.0,
                    "domain_accuracy": 0.5,  # ← ランダムレベル
                    "domain_auc": 0.5,       # ← ランダムレベル
                    "total_loss": 2.0,
                }
    
    return evaluation_step


def main(fold, device_ids, primary_device, out_dir, parallel_mode, **config):
    """メイン学習関数"""
    
    print(f"🚀 Starting DANN Training (resnet18_base)")
    print(f"Device: {primary_device}, Parallel: {parallel_mode}")
    
    # Weights & Biases初期化
    wandb.init(
        project="ResNet18_DANN_base",
        name=f"dann_base_fold{fold}" if fold is not None else "dann_base_holdout",
        config=config,
        dir=out_dir,
        tags=["dann", "resnet18_base"]
    )
    
    # CUDA環境セットアップ
    torch.cuda.set_device(primary_device)
    g = utils.setup_cuda_environment(device_ids, config['train']['seed'])
    
    # データセット読み込み
    print(f"📂 Loading datasets...")
    loader_src, loader_eval_tr, loader_eval_vl, loader_target = utils.get_datasets(config, fold, g)
    iter_target = utils.ForeverDataIterator(loader_target)
    
    print(f"  Source train batches: {len(loader_src)}")
    print(f"  Target train batches: {len(loader_target)}")
    print(f"  Source eval batches: {len(loader_eval_tr)}")
    print(f"  Source val batches: {len(loader_eval_vl)}")
    
    # モデル構築
    print(f"🏗️ Building models...")
    backbone, pre, post, func, met = utils.get_model_and_processors(config, primary_device)
    
    # バックボーン確認
    print(f"  Backbone type: {type(backbone).__name__}")
    print(f"  Has feature method: {hasattr(backbone, 'feature')}")
    
    # サンプルでテスト
    try:
        sample_batch = next(iter(loader_src))
        x_sample, y_sample = utils.safe_batch_processing(sample_batch, primary_device, pre, is_evaluation=False)
        print(f"  Sample input shape: {x_sample.shape}")
        
        with torch.no_grad():
            features = backbone.feature(x_sample[:2])  # 2サンプルでテスト
            print(f"  ✓ Backbone feature output: {features.shape}")
    except Exception as e:
        print(f"  ❌ Backbone test failed: {e}")
        return
    
    # DANN設定
    num_classes = config['model']['n_class']
    bottleneck_dim = config.get('dann', {}).get('bottleneck_dim', 256)
    domain_hidden = config.get('dann', {}).get('domain_hidden_size', 1024)
    
    print(f"📊 DANN Configuration:")
    print(f"  Num classes: {num_classes}")
    print(f"  Bottleneck dim: {bottleneck_dim}")
    print(f"  Domain hidden: {domain_hidden}")
    
    # DANNモデル作成
    classifier = DANNClassifier(backbone, num_classes, bottleneck_dim).to(primary_device)
    domain_discriminator = DomainDiscriminator(
        feature_dim=bottleneck_dim,
        hidden_dim=domain_hidden
    ).to(primary_device)
    
    # GPU並列化
    if len(device_ids) > 1 and parallel_mode == 'DataParallel':
        classifier = DataParallel(classifier, device_ids=device_ids)
        domain_discriminator = DataParallel(domain_discriminator, device_ids=device_ids)
        print(f"✓ Using DataParallel on GPUs: {device_ids}")
    
    # オプティマイザー設定
    all_params = list(classifier.parameters()) + list(domain_discriminator.parameters())
    optimizer = utils.init_optimizer(all_params, config)
    scheduler = utils.get_scheduler(optimizer, config)
    
    print(f"  Optimizer: {optimizer.__class__.__name__}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Scheduler: {scheduler.__class__.__name__ if scheduler else 'None'}")
    
    # エンジン作成
    train_step = create_train_step(classifier, domain_discriminator, optimizer, scheduler,
                                  iter_target, primary_device, config, loader_src)
    trainer = Engine(train_step)
    
    # 評価エンジン
    evaluation_step = create_evaluation_step(classifier, domain_discriminator, iter_target, primary_device, config)
    eval_tr = Engine(evaluation_step)
    eval_vl = Engine(evaluation_step)
    
    # プログレスバー
    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {'loss': f"{x['loss']:.4f}"})
    
    def safe_get(eval_dict, key, default=0.0):
        """辞書から安全にキーを取得"""
        if eval_dict and isinstance(eval_dict, dict) and key in eval_dict:
            return eval_dict[key]
        else:
            print(f"⚠️ Key '{key}' not found in evaluation result")
            return default


    # ログ処理
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        out = engine.state.output
        epoch = engine.state.epoch
        
        print(f"Epoch {epoch:3d} - "
              f"Loss: {out['loss']:.4f} "
              f"(Cls: {out['cls_loss']:.4f}, Domain: {out['domain_loss']:.4f}) | "
              f"Alpha: {out['alpha']:.4f}, Domain Acc: {out['domain_acc']:.3f}")
        
        # 評価実行（修正版）
        try:
            classifier.eval()
            domain_discriminator.eval()
            
            # 訓練データ評価（ドメイン評価含む）
            print("  🔍 Running training evaluation...")
            eval_tr.run(loader_eval_tr, max_epochs=1)
            train_eval = eval_tr.state.output
            
            # 検証データ評価（ドメイン評価含む）
            print("  🔍 Running validation evaluation...")
            eval_vl.run(loader_eval_vl, max_epochs=1)
            val_eval = eval_vl.state.output
            
            # 結果表示（ドメイン性能も含む）
            print(f"  Train - Loss: {safe_get(train_eval, 'total_loss', 1.0):.4f}, "
                  f"Cls Acc: {safe_get(train_eval, 'cls_accuracy', 0.0):.3f}, "
                  f"Domain Acc: {safe_get(train_eval, 'domain_accuracy', 0.5):.3f}")
            
            print(f"  Val   - Loss: {safe_get(val_eval, 'total_loss', 1.0):.4f}, "
                  f"Cls Acc: {safe_get(val_eval, 'cls_accuracy', 0.0):.3f}, "
                  f"Domain Acc: {safe_get(val_eval, 'domain_accuracy', 0.5):.3f}")
            
            # 拡張WandBログ
            wandb_log = {
                "epoch": epoch,
                
                # 訓練時メトリクス  
                "train/total_loss": out['loss'],
                "train/cls_loss": out['cls_loss'],
                "train/domain_loss": out['domain_loss'],
                "train/cls_acc": out['cls_acc'],
                "train/cls_auc": out['cls_auc'],
                "train/domain_acc": out['domain_acc'],
                "train/domain_auc": out['domain_auc'],
                "train/alpha": out['alpha'],
                
                # 訓練評価メトリクス
                "train_eval/total_loss": safe_get(train_eval, 'total_loss', 1.0),
                "train_eval/cls_acc": safe_get(train_eval, 'cls_accuracy', 0.0),
                "train_eval/cls_auc": safe_get(train_eval, 'cls_auc', 0.5),
                "train_eval/domain_acc": safe_get(train_eval, 'domain_accuracy', 0.5),
                "train_eval/domain_auc": safe_get(train_eval, 'domain_auc', 0.5),
                
                # 検証メトリクス  
                "val/total_loss": safe_get(val_eval, 'total_loss', 1.0),
                "val/cls_acc": safe_get(val_eval, 'cls_accuracy', 0.0),
                "val/cls_auc": safe_get(val_eval, 'cls_auc', 0.5),
                "val/domain_acc": safe_get(val_eval, 'domain_accuracy', 0.5),
                "val/domain_auc": safe_get(val_eval, 'domain_auc', 0.5),
            }
            
            # ターゲット分類性能（利用可能時）
            if train_eval and 'target_cls_accuracy' in train_eval:
                wandb_log["train_eval/target_cls_acc"] = train_eval['target_cls_accuracy']
            if val_eval and 'target_cls_accuracy' in val_eval:
                wandb_log["val/target_cls_acc"] = val_eval['target_cls_accuracy']
            
            wandb.log(wandb_log)
            
            classifier.train()
            domain_discriminator.train()
            
        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")
            # エラー時の最小限ログ
            try:
                wandb.log({
                    "epoch": epoch,
                    "train/total_loss": out['loss'],
                    "train/cls_loss": out['cls_loss'],
                    "train/domain_loss": out['domain_loss'],
                    "train/cls_acc": out['cls_acc'],
                    "train/alpha": out['alpha'],
                })
            except:
                pass
    
    # 学習実行
    try:
        max_epochs = config['train']['epoch']
        print(f"🚀 Starting training for {max_epochs} epochs")
        print("=" * 80)
        
        trainer.run(loader_src, max_epochs=max_epochs)
        
        print("=" * 80)
        print("✓ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANN Training with resnet18_base')
    parser.add_argument('--config', '-c', required=True, type=str, help='Config file path')
    parser.add_argument('--fold', '-f', type=int, default=None, help='Fold number for cross-validation')
    parser.add_argument('--device', '-d', required=True, type=str, help='CUDA device(s)')
    parser.add_argument('--parallel', '-p', choices=['DataParallel', 'single'], 
                       default='single', help='Parallelization method')
    args = parser.parse_args()
    
    # デバイス設定
    device_ids, primary_device = utils.parse_devices(args.device)
    
    # 出力ディレクトリ
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    args.out_dir = f'../logs/{config_name}_base_gpu{"_".join(map(str, device_ids))}'
    if args.fold is not None:
        args.out_dir = f"{args.out_dir}_fold{args.fold}"
    
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"📁 Output directory: {args.out_dir}")
    
    # 設定読み込み
    config = utils.load_json(args.config)
    utils.save_json(os.path.join(args.out_dir, 'config.json'), config)
    utils.command_log(args.out_dir)
    
    # メイン実行
    main(args.fold, device_ids, primary_device, args.out_dir, args.parallel, **config)
    
    # 完了フラグ
    utils.save_text(os.path.join(args.out_dir, 'finish.txt'), f'Training completed at epoch')