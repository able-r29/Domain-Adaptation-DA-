"""
DANN Training Logger - ベストモデル保存・学習再開対応版
"""

import os
import torch
import wandb
from ignite.engine import Events


def safe_get(eval_dict, key, default=0.0):
    """辞書から安全にキーを取得"""
    if eval_dict and isinstance(eval_dict, dict) and key in eval_dict:
        return eval_dict[key]
    else:
        return default


def create_logger_with_best_model_saving(trainer, classifier, domain_discriminator, eval_tr, eval_vl, 
                                        loader_eval_tr, loader_eval_vl, optimizer, scheduler, out_dir, config):
    """ベストモデル保存機能付きログ処理ハンドラーを作成"""
    
    # ベストモデル追跡変数（クロージャ内で保持）
    best_val_auc = 0.0
    best_epoch = 0
    best_model_info = {}
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        nonlocal best_val_auc, best_epoch, best_model_info
        
        out = engine.state.output
        epoch = engine.state.epoch
        
        print(f"Epoch {epoch:3d} - "
              f"Loss: {out['loss']:.4f} "
              f"(Cls: {out['cls_loss']:.4f}, Domain: {out['domain_loss']:.4f}) | "
              f"Alpha: {out['alpha']:.4f}, Domain Acc: {out['domain_acc']:.3f}")
        
        try:
            classifier.eval()
            domain_discriminator.eval()
            
            print("  🔍 Running training evaluation...")
            eval_tr.run(loader_eval_tr, max_epochs=1)
            train_eval = eval_tr.state.output
            
            print("  🔍 Running validation evaluation...")
            eval_vl.run(loader_eval_vl, max_epochs=1)
            val_eval = eval_vl.state.output
            
            print(f"  Train - Loss: {safe_get(train_eval, 'total_loss', 1.0):.4f}, "
                  f"Cls Acc: {safe_get(train_eval, 'cls_accuracy', 0.0):.3f}, "
                  f"Domain Acc: {safe_get(train_eval, 'domain_accuracy', 0.5):.3f}")
            
            print(f"  Val   - Loss: {safe_get(val_eval, 'total_loss', 1.0):.4f}, "
                  f"Cls Acc: {safe_get(val_eval, 'cls_accuracy', 0.0):.3f}, "
                  f"AUC: {safe_get(val_eval, 'cls_auc', 0.5):.3f}, "
                  f"Domain Acc: {safe_get(val_eval, 'domain_accuracy', 0.5):.3f}")
            
            # ★ ベストモデル判定（validation AUCで判定）
            current_val_auc = safe_get(val_eval, 'cls_auc', 0.0)
            
            if current_val_auc > best_val_auc:
                best_val_auc = current_val_auc
                best_epoch = epoch
                
                print(f"  🏆 NEW BEST MODEL! Val AUC: {current_val_auc:.4f} (previous: {best_val_auc:.4f})")
                
                # ベストモデル情報を記録
                best_model_info = {
                    'epoch': epoch,
                    'val_auc': current_val_auc,
                    'val_acc': safe_get(val_eval, 'cls_accuracy', 0.0),
                    'domain_acc': safe_get(val_eval, 'domain_accuracy', 0.5),
                    'train_metrics': out,
                    'val_metrics': val_eval,
                }
                
                # ★ ベストモデル保存
                best_model_dict = {
                    'epoch': epoch,
                    'best_val_auc': current_val_auc,
                    'val_cls_acc': safe_get(val_eval, 'cls_accuracy', 0.0),
                    'val_domain_acc': safe_get(val_eval, 'domain_accuracy', 0.5),
                    'classifier_state_dict': classifier.state_dict(),
                    'domain_discriminator_state_dict': domain_discriminator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'config': config,
                    'train_metrics': out,
                    'val_metrics': val_eval,
                    'wandb_id': wandb.run.id if wandb.run else None,
                }
                
                best_model_path = os.path.join(out_dir, 'best_model.pth')
                torch.save(best_model_dict, best_model_path)
                print(f"    💾 Saved: best_model.pth")
            
            # 現在のベスト情報を表示
            print(f"  📊 Best: AUC {best_val_auc:.4f} at epoch {best_epoch}")
            
            # WandBログ（ベスト情報追加）
            wandb_log = {
                "epoch": epoch,
                "train/total_loss": out['loss'],
                "train/cls_loss": out['cls_loss'],
                "train/domain_loss": out['domain_loss'],
                "train/cls_acc": out['cls_acc'],
                "train/cls_auc": out['cls_auc'],
                "train/domain_acc": out['domain_acc'],
                "train/domain_auc": out['domain_auc'],
                "train/alpha": out['alpha'],
                "train_eval/total_loss": safe_get(train_eval, 'total_loss', 1.0),
                "train_eval/cls_acc": safe_get(train_eval, 'cls_accuracy', 0.0),
                "train_eval/cls_auc": safe_get(train_eval, 'cls_auc', 0.5),
                "train_eval/domain_acc": safe_get(train_eval, 'domain_accuracy', 0.5),
                "train_eval/domain_auc": safe_get(train_eval, 'domain_auc', 0.5),
                "val/total_loss": safe_get(val_eval, 'total_loss', 1.0),
                "val/cls_acc": safe_get(val_eval, 'cls_accuracy', 0.0),
                "val/cls_auc": safe_get(val_eval, 'cls_auc', 0.5),
                "val/domain_acc": safe_get(val_eval, 'domain_accuracy', 0.5),
                "val/domain_auc": safe_get(val_eval, 'domain_auc', 0.5),
                # ★ ベスト情報追加
                "best/val_auc": best_val_auc,
                "best/epoch": best_epoch,
                "best/val_acc": best_model_info.get('val_acc', 0.0),
                "best/domain_acc": best_model_info.get('domain_acc', 0.5),
            }
            
            # オプション追加
            for eval_data, prefix in [(train_eval, "train_eval"), (val_eval, "val")]:
                if eval_data and 'target_cls_accuracy' in eval_data:
                    wandb_log[f"{prefix}/target_cls_acc"] = eval_data['target_cls_accuracy']
            
            wandb.log(wandb_log)
            
            classifier.train()
            domain_discriminator.train()
            
        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")
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
    
    # ★ 自動チェックポイント保存（中断対策）
    @trainer.on(Events.EPOCH_COMPLETED)
    def auto_save_checkpoint(engine):
        nonlocal best_val_auc, best_epoch, best_model_info
        
        epoch = engine.state.epoch
        
        # 最新チェックポイント（毎エポック更新）
        checkpoint_dict = {
            'epoch': epoch,
            'classifier_state_dict': classifier.state_dict(),
            'domain_discriminator_state_dict': domain_discriminator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': config,
            'best_val_auc': best_val_auc,
            'best_epoch': best_epoch,
            'best_model_info': best_model_info,
            'wandb_id': wandb.run.id if wandb.run else None,
        }
        
        # 最新チェックポイント保存
        latest_path = os.path.join(out_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint_dict, latest_path)
        
        # 10エポックごとに番号付きチェックポイント
        if epoch % 10 == 0:
            numbered_path = os.path.join(out_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint_dict, numbered_path)
            print(f"  💾 Checkpoint: epoch_{epoch}.pth")
    
    # ★ ベストモデル情報を返す関数（トレーニング後に使用）
    def get_best_model_info():
        return {
            'best_val_auc': best_val_auc,
            'best_epoch': best_epoch,
            'best_model_info': best_model_info
        }
    
    return get_best_model_info


def create_logger(trainer, classifier, domain_discriminator, eval_tr, eval_vl, 
                 loader_eval_tr, loader_eval_vl):
    """従来のログ処理ハンドラー（後方互換性用）"""
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        out = engine.state.output
        epoch = engine.state.epoch
        
        print(f"Epoch {epoch:3d} - "
              f"Loss: {out['loss']:.4f} "
              f"(Cls: {out['cls_loss']:.4f}, Domain: {out['domain_loss']:.4f}) | "
              f"Alpha: {out['alpha']:.4f}, Domain Acc: {out['domain_acc']:.3f}")
        
        try:
            classifier.eval()
            domain_discriminator.eval()
            
            print("  🔍 Running training evaluation...")
            eval_tr.run(loader_eval_tr, max_epochs=1)
            train_eval = eval_tr.state.output
            
            print("  🔍 Running validation evaluation...")
            eval_vl.run(loader_eval_vl, max_epochs=1)
            val_eval = eval_vl.state.output
            
            print(f"  Train - Loss: {safe_get(train_eval, 'total_loss', 1.0):.4f}, "
                  f"Cls Acc: {safe_get(train_eval, 'cls_accuracy', 0.0):.3f}, "
                  f"Domain Acc: {safe_get(train_eval, 'domain_accuracy', 0.5):.3f}")
            
            print(f"  Val   - Loss: {safe_get(val_eval, 'total_loss', 1.0):.4f}, "
                  f"Cls Acc: {safe_get(val_eval, 'cls_accuracy', 0.0):.3f}, "
                  f"Domain Acc: {safe_get(val_eval, 'domain_accuracy', 0.5):.3f}")
            
            # WandBログ
            wandb_log = {
                "epoch": epoch,
                "train/total_loss": out['loss'],
                "train/cls_loss": out['cls_loss'],
                "train/domain_loss": out['domain_loss'],
                "train/cls_acc": out['cls_acc'],
                "train/cls_auc": out['cls_auc'],
                "train/domain_acc": out['domain_acc'],
                "train/domain_auc": out['domain_auc'],
                "train/alpha": out['alpha'],
                "train_eval/total_loss": safe_get(train_eval, 'total_loss', 1.0),
                "train_eval/cls_acc": safe_get(train_eval, 'cls_accuracy', 0.0),
                "train_eval/cls_auc": safe_get(train_eval, 'cls_auc', 0.5),
                "train_eval/domain_acc": safe_get(train_eval, 'domain_accuracy', 0.5),
                "train_eval/domain_auc": safe_get(train_eval, 'domain_auc', 0.5),
                "val/total_loss": safe_get(val_eval, 'total_loss', 1.0),
                "val/cls_acc": safe_get(val_eval, 'cls_accuracy', 0.0),
                "val/cls_auc": safe_get(val_eval, 'cls_auc', 0.5),
                "val/domain_acc": safe_get(val_eval, 'domain_accuracy', 0.5),
                "val/domain_auc": safe_get(val_eval, 'domain_auc', 0.5),
            }
            
            # オプション追加
            for eval_data, prefix in [(train_eval, "train_eval"), (val_eval, "val")]:
                if eval_data and 'target_cls_accuracy' in eval_data:
                    wandb_log[f"{prefix}/target_cls_acc"] = eval_data['target_cls_accuracy']
            
            wandb.log(wandb_log)
            
            classifier.train()
            domain_discriminator.train()
            
        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")
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


def setup_wandb(fold, out_dir, config, resume_id=None):
    """WandB初期化（再開対応）"""
    wandb.init(
        project="ResNet18_DANN_base",
        name=f"dann_base_fold{fold}" if fold is not None else "dann_base_holdout",
        config=config,
        dir=out_dir,
        tags=["dann", "resnet18_base"],
        resume="allow" if resume_id else None,
        id=resume_id
    )


def print_system_info(device_ids, primary_device, parallel_mode):
    """システム情報表示"""
    print(f"🚀 Starting DANN Training (resnet18_base)")
    print(f"Device: {primary_device}, Parallel: {parallel_mode}")


def print_dataset_info(loader_src, loader_target, loader_eval_tr, loader_eval_vl):
    """データセット情報表示"""
    print(f"📂 Loading datasets...")
    print(f"  Source train batches: {len(loader_src)}")
    print(f"  Target train batches: {len(loader_target)}")
    print(f"  Source eval batches: {len(loader_eval_tr)}")
    print(f"  Source val batches: {len(loader_eval_vl)}")


def print_model_info(backbone, num_classes, bottleneck_dim, domain_hidden):
    """モデル情報表示"""
    print(f"🏗️ Building models...")
    print(f"  Backbone type: {type(backbone).__name__}")
    print(f"  Has feature method: {hasattr(backbone, 'feature')}")
    print(f"📊 DANN Configuration:")
    print(f"  Num classes: {num_classes}")
    print(f"  Bottleneck dim: {bottleneck_dim}")
    print(f"  Domain hidden: {domain_hidden}")


def print_optimizer_info(optimizer, scheduler):
    """オプティマイザー情報表示"""
    print(f"  Optimizer: {optimizer.__class__.__name__}")
    print(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"  Scheduler: {scheduler.__class__.__name__ if scheduler else 'None'}")