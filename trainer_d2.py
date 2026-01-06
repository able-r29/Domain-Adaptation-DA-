"""
DANN Training Script - ベストモデル保存・学習再開・GPU並列対応版
"""

import argparse
import os
import torch
from torch.nn.parallel import DataParallel
from ignite.engine import Engine
from ignite.contrib.handlers import ProgressBar

from domain_discriminator import DomainDiscriminator, DANNClassifier
from train_engine import create_train_step, create_evaluation_step
from train_logger import create_logger_with_best_model_saving, setup_wandb, print_system_info, print_model_info, print_optimizer_info
import utils


def validate_backbone(backbone, loader_src, primary_device, pre):
    """バックボーン検証"""
    try:
        sample_batch = next(iter(loader_src))
        x_sample, y_sample = utils.safe_batch_processing(sample_batch, primary_device, pre, is_evaluation=False)
        print(f"  Sample input shape: {x_sample.shape}")
        
        with torch.no_grad():
            features = backbone.feature(x_sample[:2])
            print(f"  ✓ Backbone feature output: {features.shape}")
        return True
    except Exception as e:
        print(f"  ❌ Backbone test failed: {e}")
        return False


def load_checkpoint_if_exists(out_dir, resume_path=None):
    """チェックポイントを自動検出・読み込み"""
    if resume_path and os.path.exists(resume_path):
        print(f"📄 Loading specified checkpoint: {resume_path}")
        return torch.load(resume_path, map_location='cpu'), resume_path
    
    # 自動検出：interrupted > latest > なし
    interrupted_path = os.path.join(out_dir, 'interrupted_checkpoint.pth')
    latest_path = os.path.join(out_dir, 'latest_checkpoint.pth')
    
    if os.path.exists(interrupted_path):
        print(f"🔄 Auto-detected interrupted training: {interrupted_path}")
        return torch.load(interrupted_path, map_location='cpu'), interrupted_path
    elif os.path.exists(latest_path):
        print(f"🔄 Auto-detected latest checkpoint: {latest_path}")
        return torch.load(latest_path, map_location='cpu'), latest_path
    else:
        return None, None


def restore_model_states(checkpoint, classifier, domain_discriminator, optimizer, scheduler):
    """モデル・オプティマイザーの状態を復元"""
    try:
        # DataParallel対応の状態復元
        classifier_state = checkpoint['classifier_state_dict']
        domain_state = checkpoint['domain_discriminator_state_dict']
        
        # DataParallelのstate_dictは'module.'プレフィックスを持つ場合がある
        if hasattr(classifier, 'module'):
            # 現在がDataParallelで、保存もDataParallelの場合
            classifier.load_state_dict(classifier_state)
            domain_discriminator.load_state_dict(domain_state)
        else:
            # 現在が単一GPUで、保存がDataParallelの場合は'module.'を除去
            if any(k.startswith('module.') for k in classifier_state.keys()):
                classifier_state = {k.replace('module.', ''): v for k, v in classifier_state.items()}
            if any(k.startswith('module.') for k in domain_state.keys()):
                domain_state = {k.replace('module.', ''): v for k, v in domain_state.items()}
            
            classifier.load_state_dict(classifier_state)
            domain_discriminator.load_state_dict(domain_state)
        
        # オプティマイザー復元
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # スケジューラー復元
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  ✓ Restored: classifier, domain_discriminator, optimizer, scheduler")
        else:
            print("  ✓ Restored: classifier, domain_discriminator, optimizer")
            
        return True
    except Exception as e:
        print(f"  ⚠️ Weight restoration failed: {e}")
        print("  Continuing with random initialization...")
        return False


def setup_parallel_training(classifier, domain_discriminator, device_ids, primary_device, parallel_mode):
    """並列学習のセットアップ"""
    
    # GPU情報表示
    print(f"\n🔍 GPU Information:")
    for device_id in device_ids:
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(device_id)
            total_mem = props.total_memory / 1e9
            print(f"  GPU {device_id}: {props.name}, {total_mem:.2f} GB")
        else:
            print(f"  GPU {device_id}: Not available")
    
    # 並列化の設定
    if len(device_ids) > 1:
        if parallel_mode == 'DataParallel':
            print(f"\n🚀 Setting up DataParallel training:")
            print(f"  Primary device: cuda:{device_ids[0]}")
            print(f"  All devices: {device_ids}")
            
            # DataParallel適用
            classifier = DataParallel(classifier, device_ids=device_ids)
            domain_discriminator = DataParallel(domain_discriminator, device_ids=device_ids)
            
            print(f"  ✓ DataParallel enabled on GPUs: {device_ids}")
            
            # メモリ使用量表示
            utils.print_gpu_memory_info(device_ids)
            
        else:
            print(f"⚠️ Multiple GPUs detected but parallel mode is '{parallel_mode}'")
            print(f"  Using single GPU: cuda:{device_ids[0]}")
            
    else:
        print(f"🔧 Single GPU training on cuda:{device_ids[0]}")
        utils.print_gpu_memory_info(device_ids)
    
    return classifier, domain_discriminator


def adjust_batch_size_for_parallel(config, device_ids):
    """並列処理用にバッチサイズを調整"""
    if len(device_ids) > 1:
        original_batch_size = config['dataset']['batch_size']
        # 各GPUに配分されるバッチサイズを考慮
        per_gpu_batch_size = original_batch_size // len(device_ids)
        
        if per_gpu_batch_size < 8:
            print(f"⚠️ Batch size per GPU ({per_gpu_batch_size}) is very small!")
            print(f"  Consider increasing total batch size or reducing number of GPUs")
        
        print(f"📊 Batch size configuration:")
        print(f"  Total batch size: {original_batch_size}")
        print(f"  Per GPU batch size: {per_gpu_batch_size}")
        print(f"  Number of GPUs: {len(device_ids)}")
    
    return config


def main(fold, device_ids, primary_device, out_dir, parallel_mode, resume_path=None, **config):
    """メイン学習関数（完全修正版）"""
    
    print_system_info(device_ids, primary_device, parallel_mode)
    
    # バッチサイズ調整
    config = adjust_batch_size_for_parallel(config, device_ids)
    
    # 学習再開チェック
    checkpoint, used_checkpoint_path = load_checkpoint_if_exists(out_dir, resume_path)
    
    if checkpoint:
        start_epoch = checkpoint['epoch'] + 1
        config = checkpoint['config']
        resume_wandb_id = checkpoint.get('wandb_id')
        best_val_auc = checkpoint.get('best_val_auc', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0)
        print(f"  📊 Resuming from epoch {start_epoch}")
        print(f"  📈 Previous best: AUC {best_val_auc:.4f} at epoch {best_epoch}")
    else:
        start_epoch = 1
        resume_wandb_id = None
        print(f"  📊 Starting fresh training from epoch 1")
    
    # WandB初期化
    setup_wandb(fold, out_dir, config, resume_wandb_id)
    
    # 環境・データセットセットアップ
    torch.cuda.set_device(primary_device)
    g = utils.setup_cuda_environment(device_ids, config['train']['seed'])
    
    # ★ データセット取得（修正版：4つの戻り値）
    loader_src_train, loader_src_val, loader_target_train, loader_target_val = utils.get_datasets(config, fold, g)
    
    # 学習用イテレータ
    iter_target_train = utils.ForeverDataIterator(loader_target_train)
    
    # 評価用イテレータ
    iter_target_val = utils.ForeverDataIterator(loader_target_val)
    
    # モデル構築
    backbone, pre, post, func, met = utils.get_model_and_processors(config, primary_device)
    if not validate_backbone(backbone, loader_src_train, primary_device, pre):
        return
    
    # DANN設定
    num_classes = config['model']['n_class']
    bottleneck_dim = config.get('dann', {}).get('bottleneck_dim', 256)
    domain_hidden = config.get('dann', {}).get('domain_hidden_size', 1024)
    print_model_info(backbone, num_classes, bottleneck_dim, domain_hidden)
    
    # モデル作成
    classifier = DANNClassifier(backbone, num_classes, bottleneck_dim).to(primary_device)
    domain_discriminator = DomainDiscriminator(feature_dim=bottleneck_dim, hidden_dim=domain_hidden).to(primary_device)
    
    # 並列化セットアップ
    classifier, domain_discriminator = setup_parallel_training(
        classifier, domain_discriminator, device_ids, primary_device, parallel_mode
    )
    
    # オプティマイザー
    all_params = list(classifier.parameters()) + list(domain_discriminator.parameters())
    optimizer = utils.init_optimizer(all_params, config)
    scheduler = utils.get_scheduler(optimizer, config)
    print_optimizer_info(optimizer, scheduler)
    
    # 学習済み状態の復元（再開時）
    if checkpoint:
        print("📥 Restoring model states...")
        restore_model_states(checkpoint, classifier, domain_discriminator, optimizer, scheduler)
    
    # エンジン作成
    train_step = create_train_step(
        classifier, domain_discriminator, optimizer, scheduler,
        iter_target_train, primary_device, config, loader_src_train, start_epoch
    )
    trainer = Engine(train_step)
    
    evaluation_step = create_evaluation_step(
        classifier, domain_discriminator, iter_target_val, primary_device, config, start_epoch
    )
    evaluator = Engine(evaluation_step)
    
    # プログレスバー
    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {'loss': f"{x['loss']:.4f}"})
    
    # ベストモデル保存機能付きログ
    get_best_model_info = create_logger_with_best_model_saving(
        trainer, classifier, domain_discriminator, evaluator, 
        loader_src_val, optimizer, scheduler, out_dir, config
    )
    
    # 学習実行
    try:
        max_epochs = config['train']['epoch']
        
        if start_epoch > max_epochs:
            print(f"📄 Training already completed (epoch {start_epoch-1}/{max_epochs})")
            return
        
        print(f"🚀 Starting training from epoch {start_epoch} to {max_epochs}")
        print(f"  Remaining epochs: {max_epochs - start_epoch + 1}")
        print("=" * 80)
        
        # エンジンの状態を調整（再開時）
        if start_epoch > 1:
            trainer.state.epoch = start_epoch - 1
            trainer.state.iteration = (start_epoch - 1) * len(loader_src_train)
        
        remaining_epochs = max_epochs - start_epoch + 1
        trainer.run(loader_src_train, max_epochs=remaining_epochs)
        
        print("=" * 80)
        print("✓ Training completed successfully!")
        
        # 最終モデル保存
        final_model_dict = {
            'epoch': max_epochs,
            'classifier_state_dict': classifier.state_dict(),
            'domain_discriminator_state_dict': domain_discriminator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'final_metrics': trainer.state.output,
        }
        
        final_model_path = os.path.join(out_dir, 'final_model.pth')
        torch.save(final_model_dict, final_model_path)
        print(f"💾 Final model saved: final_model.pth")
        
        # ベストモデル情報の最終表示
        best_info = get_best_model_info()
        print(f"\n🏆 TRAINING SUMMARY:")
        print(f"  Best Validation AUC: {best_info['best_val_auc']:.4f}")
        print(f"  Best Epoch: {best_info['best_epoch']}")
        print(f"  Best Model Path: {os.path.join(out_dir, 'best_model.pth')}")
        
        if best_info['best_model_info']:
            info = best_info['best_model_info']
            print(f"  Best Val Acc: {info.get('val_acc', 0.0):.3f}")
            print(f"  Best Domain Acc: {info.get('domain_acc', 0.5):.3f}")
        
        # GPU使用率サマリー
        print(f"\n📊 Training completed on {len(device_ids)} GPU(s): {device_ids}")
        utils.print_gpu_memory_info(device_ids)
        
    except KeyboardInterrupt:
        print(f"\n🛑 Training interrupted at epoch {trainer.state.epoch}")
        
        # 中断時自動保存
        interrupt_dict = {
            'epoch': trainer.state.epoch,
            'classifier_state_dict': classifier.state_dict(),
            'domain_discriminator_state_dict': domain_discriminator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': config,
            'best_val_auc': get_best_model_info()['best_val_auc'],
            'best_epoch': get_best_model_info()['best_epoch'],
            'best_model_info': get_best_model_info()['best_model_info'],
            'wandb_id': wandb.run.id if wandb.run else None,
        }
        interrupt_path = os.path.join(out_dir, 'interrupted_checkpoint.pth')
        torch.save(interrupt_dict, interrupt_path)
        print(f"💾 Interrupted state saved: interrupted_checkpoint.pth")
        
        # 再開コマンド表示
        device_str = ','.join(map(str, device_ids))
        resume_cmd = f"python trainer_d2.py --config {args.config} --device {device_str}"
        if args.fold is not None:
            resume_cmd += f" --fold {args.fold}"
        if args.parallel != 'single':
            resume_cmd += f" --parallel {args.parallel}"
        print(f"  To resume: {resume_cmd}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DANN Training with Multi-GPU Support, Best Model Saving and Resume')
    parser.add_argument('--config', '-c', required=True, type=str, help='Config file path')
    parser.add_argument('--fold', '-f', type=int, default=None, help='Fold number for cross-validation')
    parser.add_argument('--device', '-d', required=True, type=str, 
                       help='CUDA device(s). Single: "0", Multiple: "0,1,2,3"')  # ← 説明更新
    parser.add_argument('--parallel', '-p', choices=['DataParallel', 'single'], 
                       default='DataParallel', help='Parallelization method')  # ← デフォルトをDataParallelに変更
    parser.add_argument('--resume', '-r', type=str, default=None, 
                       help='Resume from specific checkpoint file')
    args = parser.parse_args()
    
    # ★ 複数GPU対応のデバイス解析
    device_ids, primary_device = utils.parse_devices(args.device)
    
    # デバイス情報表示
    print(f"🔧 Device Configuration:")
    print(f"  Specified devices: {args.device}")
    print(f"  Parsed device IDs: {device_ids}")
    print(f"  Primary device: {primary_device}")
    print(f"  Parallel mode: {args.parallel}")
    
    # 利用可能なGPUチェック
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        exit(1)
    
    available_gpus = list(range(torch.cuda.device_count()))
    invalid_gpus = [gpu_id for gpu_id in device_ids if gpu_id not in available_gpus]
    if invalid_gpus:
        print(f"❌ Invalid GPU IDs: {invalid_gpus}")
        print(f"   Available GPUs: {available_gpus}")
        exit(1)
    
    # 出力ディレクトリ（複数GPU対応）
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    device_str = "_".join(map(str, device_ids))
    args.out_dir = f'../dann_exp_final2'
    if args.fold is not None:
        args.out_dir = f"{args.out_dir}_fold{args.fold}"
    
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"📁 Output directory: {args.out_dir}")
    
    # 設定読み込み（再開時は保存された設定を使用）
    checkpoint, _ = load_checkpoint_if_exists(args.out_dir, args.resume)
    if checkpoint:
        config = checkpoint['config']
        print(f"📄 Using config from checkpoint")
    else:
        config = utils.load_json(args.config)
        utils.save_json(os.path.join(args.out_dir, 'config.json'), config)
        utils.command_log(args.out_dir)
    
    # 実行
    main(args.fold, device_ids, primary_device, args.out_dir, args.parallel, 
         resume_path=args.resume, **config)
    
    # 完了フラグ
    utils.save_text(os.path.join(args.out_dir, 'finish.txt'), f'Training completed')