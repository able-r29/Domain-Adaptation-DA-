"""
DANN Training Logger - 簡潔版（train/validationのみ）
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


def create_logger_with_best_model_saving(trainer, classifier, domain_discriminator, evaluator, 
                                        loader_val, optimizer, scheduler, out_dir, config):
    """ベストモデル保存機能付きログ処理ハンドラー（簡潔版）"""
    
    best_val_auc = 0.0
    best_epoch = 0
    best_model_info = {}
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        nonlocal best_val_auc, best_epoch, best_model_info
        
        # Train時のメトリクス
        train_out = engine.state.output
        epoch = train_out.get('actual_epoch', engine.state.epoch)
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}")
        print(f"{'='*80}")
        
        # ★ Train時のログ表示
        print(f"📈 Training Metrics:")
        print(f"  Total Loss: {train_out['loss']:.4f}")
        print(f"  Classifier Loss: {train_out['cls_loss']:.4f} | Acc: {train_out['cls_acc']:.3f} | AUC: {train_out['cls_auc']:.3f}")
        print(f"  Domain Loss: {train_out['domain_loss']:.4f} | Acc: {train_out['domain_acc']:.3f} | AUC: {train_out['domain_auc']:.3f}")
        print(f"  Alpha: {train_out['alpha']:.4f} (p={train_out.get('p', 0.0):.4f})")
        
        try:
            classifier.eval()
            domain_discriminator.eval()
            
            # ★ Validation評価
            print(f"\n🔍 Running validation...")
            evaluator.run(loader_val, max_epochs=1)
            val_out = evaluator.state.output
            
            # ★ Validation時のログ表示
            print(f"📊 Validation Metrics:")
            print(f"  Total Loss: {safe_get(val_out, 'total_loss', 1.0):.4f}")
            print(f"  Classifier Loss: {safe_get(val_out, 'cls_loss', 1.0):.4f} | Acc: {safe_get(val_out, 'cls_accuracy', 0.0):.3f} | AUC: {safe_get(val_out, 'cls_auc', 0.5):.3f}")
            print(f"  Domain Loss: {safe_get(val_out, 'domain_loss', 1.0):.4f} | Acc: {safe_get(val_out, 'domain_accuracy', 0.5):.3f} | AUC: {safe_get(val_out, 'domain_auc', 0.5):.3f}")
            
            # ★ ベストモデル判定（validation AUCで判定）
            current_val_auc = safe_get(val_out, 'cls_auc', 0.0)
            
            if current_val_auc > best_val_auc:
                best_val_auc = current_val_auc
                best_epoch = epoch
                
                print(f"\n🏆 NEW BEST MODEL! Val AUC: {current_val_auc:.4f}")
                
                best_model_info = {
                    'epoch': epoch,
                    'val_auc': current_val_auc,
                    'val_acc': safe_get(val_out, 'cls_accuracy', 0.0),
                    'domain_acc': safe_get(val_out, 'domain_accuracy', 0.5),
                    'train_metrics': train_out,
                    'val_metrics': val_out,
                }
                
                # ベストモデル保存
                best_model_dict = {
                    'epoch': epoch,
                    'best_val_auc': current_val_auc,
                    'val_cls_acc': safe_get(val_out, 'cls_accuracy', 0.0),
                    'val_domain_acc': safe_get(val_out, 'domain_accuracy', 0.5),
                    'classifier_state_dict': classifier.state_dict(),
                    'domain_discriminator_state_dict': domain_discriminator.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'config': config,
                    'train_metrics': train_out,
                    'val_metrics': val_out,
                    'wandb_id': wandb.run.id if wandb.run else None,
                }
                
                best_model_path = os.path.join(out_dir, 'best_model.pth')
                torch.save(best_model_dict, best_model_path)
                print(f"  💾 Saved: best_model.pth")
            
            print(f"\n📈 Best Model: AUC {best_val_auc:.4f} at epoch {best_epoch}")
            
            # ★ WandBログ（必要なメトリクスのみ）
            wandb_log = {
                "epoch": epoch,
                
                # Train時のメトリクス
                "train/total_loss": train_out['loss'],
                "train/cls_loss": train_out['cls_loss'],
                "train/cls_acc": train_out['cls_acc'],
                "train/cls_auc": train_out['cls_auc'],
                "train/domain_loss": train_out['domain_loss'],
                "train/domain_acc": train_out['domain_acc'],
                "train/domain_auc": train_out['domain_auc'],
                "train/alpha": train_out['alpha'],
                
                # Validation時のメトリクス
                "val/total_loss": safe_get(val_out, 'total_loss', 1.0),
                "val/cls_loss": safe_get(val_out, 'cls_loss', 1.0),
                "val/cls_acc": safe_get(val_out, 'cls_accuracy', 0.0),
                "val/cls_auc": safe_get(val_out, 'cls_auc', 0.5),
                "val/domain_loss": safe_get(val_out, 'domain_loss', 1.0),
                "val/domain_acc": safe_get(val_out, 'domain_accuracy', 0.5),
                "val/domain_auc": safe_get(val_out, 'domain_auc', 0.5),
                
                # ベストモデル情報
                "best/val_auc": best_val_auc,
                "best/epoch": best_epoch,
            }
            
            wandb.log(wandb_log)
            
            classifier.train()
            domain_discriminator.train()
            
        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def auto_save_checkpoint(engine):
        nonlocal best_val_auc, best_epoch, best_model_info
        
        epoch = engine.state.output.get('actual_epoch', engine.state.epoch)
        
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
        
        latest_path = os.path.join(out_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint_dict, latest_path)
        
        if epoch % 10 == 0:
            numbered_path = os.path.join(out_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint_dict, numbered_path)
            print(f"  💾 Checkpoint: epoch_{epoch}.pth")
    
    def get_best_model_info():
        return {
            'best_val_auc': best_val_auc,
            'best_epoch': best_epoch,
            'best_model_info': best_model_info
        }
    
    return get_best_model_info


def setup_wandb(fold, out_dir, config, resume_id=None):
    """WandB初期化"""
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