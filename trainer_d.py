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
    """学習ステップ関数（デバッグ出力を削減）"""
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
            # データ処理（学習時）
            x_s, y_s = utils.safe_batch_processing(batch, device, pre, is_evaluation=False)
            target_batch = next(iter_target)
            x_t, _ = utils.safe_batch_processing(target_batch, device, pre, is_evaluation=False)
            
            # デバッグ情報（最初の2エポック、5イテレーションごと）
            if engine.state.epoch <= 2 and engine.state.iteration % 5 == 0:
                print(f"📝 Train Epoch {engine.state.epoch}, Iter {engine.state.iteration}")
                print(f"  Source batch size: {x_s.shape[0] if hasattr(x_s, 'shape') else len(x_s) if hasattr(x_s, '__len__') else 'unknown'}")
                print(f"  Target batch size: {x_t.shape[0] if hasattr(x_t, 'shape') else len(x_t) if hasattr(x_t, '__len__') else 'unknown'}")
            
            # GRL強度調整
            alpha = 1.0
            utils.set_alpha_safely(domain_discriminator, alpha)
            
            # 分類器で特徴抽出と分類
            y_s_pred, f_s = classifier(x_s)
            _, f_t = classifier(x_t)
            
            # 分類損失
            cls_loss = cls_criterion(y_s_pred, y_s)
            
            # ドメイン損失
            domain_pred_s = domain_discriminator(f_s)
            domain_pred_t = domain_discriminator(f_t)
            
            domain_label_s = torch.zeros(f_s.size(0), 1).to(device)
            domain_label_t = torch.ones(f_t.size(0), 1).to(device)
            
            domain_loss = (domain_criterion(domain_pred_s, domain_label_s) + 
                          domain_criterion(domain_pred_t, domain_label_t))
            
            # 総損失
            total_loss = cls_loss + trade_off * domain_loss
            
            # バックプロパゲーション
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            # メトリクス計算
            with torch.no_grad():
                domain_acc_s = ((domain_pred_s < 0.5).float() == domain_label_s).float().mean()
                domain_acc_t = ((domain_pred_t >= 0.5).float() == domain_label_t).float().mean()
                domain_acc = (domain_acc_s + domain_acc_t) / 2
            
            return {
                "loss": total_loss.item(),
                "cls_loss": cls_loss.item(),
                "domain_loss": domain_loss.item(),
                "domain_acc": domain_acc.item(),
                "alpha": alpha
            }
            
        except Exception as e:
            print(f"❌ Training step failed: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    return train_step


def create_evaluation_step(classifier, device, config):
    """評価ステップ関数（修正版：前処理必須）"""
    # 前処理関数を事前に取得
    _, pre, _, _, _ = utils.get_model_and_processors(config, device)
    
    if pre is None:
        print("⚠️ Warning: No preprocessing function found!")
    
    def evaluation_step(engine, batch):
        classifier.eval()
        with torch.no_grad():
            try:
                # 評価時も前処理を必ず適用
                x, y = utils.safe_batch_processing(batch, device, pre, is_evaluation=True)
                
                # デバッグ出力（最初の評価時のみ）
                if engine.state.iteration == 1:
                    print(f"🔍 Evaluation input check:")
                    if isinstance(x, dict):
                        for k, v in x.items():
                            if hasattr(v, 'shape'):
                                print(f"  {k}: {v.shape}")
                    else:
                        print(f"  x shape: {x.shape}")
                    print(f"  y shape: {y.shape}")
                
                # 予測
                y_pred, _ = classifier(x)
                
                # サイズ確認
                if y_pred.shape[0] != y.shape[0]:
                    print(f"❌ Batch size mismatch: pred {y_pred.shape}, label {y.shape}")
                    return {"accuracy": 0.0}
                
                # 精度計算
                correct = (y_pred.argmax(dim=1) == y).float().mean()
                
                return {"accuracy": correct.item()}
                    
            except Exception as e:
                print(f"❌ Evaluation step failed: {e}")
                print(f"Batch type: {type(batch)}")
                if isinstance(batch, (list, tuple)):
                    print(f"Batch length: {len(batch)}")
                    for i, item in enumerate(batch):
                        if hasattr(item, 'shape'):
                            print(f"  batch[{i}] shape: {item.shape}")
                        else:
                            print(f"  batch[{i}] type: {type(item)}")
                
                import traceback
                traceback.print_exc()
                return {"accuracy": 0.0}
    
    return evaluation_step


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
    
    # 評価エンジン（修正版）
    evaluation_step = create_evaluation_step(classifier, primary_device, config)
    eval_tr = Engine(evaluation_step)
    eval_vl = Engine(evaluation_step)
    
    # プログレスバー
    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {'loss': x['loss']})
    
    # ログ処理
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        out = engine.state.output
        
        # 基本メトリクス
        metrics = {
            "epoch": engine.state.epoch,
            "train/loss": out['loss'],
            "train/cls_loss": out['cls_loss'],
            "train/domain_loss": out['domain_loss'],
            "train/domain_acc": out['domain_acc'],
            "train/alpha": out['alpha']
        }
        
        wandb.log(metrics)
        
        print(f"Epoch {engine.state.epoch:3d} - "
              f"Loss: {out['loss']:.4f} "
              f"(Cls: {out['cls_loss']:.4f}, Domain: {out['domain_loss']:.4f}, "
              f"Domain Acc: {out['domain_acc']:.3f}, α: {out['alpha']:.3f})")
        
        # 評価実行（5エポックごと）
        if engine.state.epoch % 5 == 0:
            classifier.eval()
            try:
                print("🔍 Running evaluation...")
                
                # Train評価
                eval_tr.run(loader_eval_tr, max_epochs=1)
                train_acc = eval_tr.state.output.get("accuracy", 0.0)
                
                # Validation評価
                eval_vl.run(loader_eval_vl, max_epochs=1)
                val_acc = eval_vl.state.output.get("accuracy", 0.0)
                
                eval_metrics = {
                    "train/accuracy": train_acc,
                    "val/accuracy": val_acc
                }
                wandb.log(eval_metrics)
                
                print(f"📊 Epoch {engine.state.epoch} - Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
                
            except Exception as e:
                print(f"⚠️ Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
            
            classifier.train()
    
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