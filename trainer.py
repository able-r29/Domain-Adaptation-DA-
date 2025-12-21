import argparse
import json
import os
import pathlib
import random
import sys

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.handlers.tensorboard_logger import (
    GradsHistHandler,
    GradsScalarHandler,
    TensorboardLogger,
    WeightsHistHandler,
    WeightsScalarHandler,
    global_step_from_engine,
)
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger
from ignite.engine import Engine, Events
from torch.nn import Module
from torch.optim.optimizer import Optimizer

import wandb

import common.parse_optuna as parse_optuna
import common.early_stopping as early_stopping
import common.exponential_moving_average as EMA
import datasets.datasets as dataset
import models.models as models
import utils
import tqdm
from PIL import Image, ImageFile
import collections
import pprint

from domain_discriminator import DomainDiscriminator
from dann import DomainAdversarialLoss, ImageClassifier
from data_utils import ForeverDataIterator

matplotlib.use('Agg')

def init_opt(params, name:str, **karg):
    """
    Optimizerの初期化（パラメータリストを直接受け取るように修正）
    """
    opt_cls = {
        'sgd'     : optim.SGD,
        'rmsprop' : optim.RMSprop,
        'adam'    : optim.Adam
    }[name]

    if name == 'sgd' and 'momentum' in karg:
        karg['nesterov'] = True

    dst = opt_cls(params, **karg)
    return dst

def main(i_fold:int, device:str, out_dir:str, **kargs) -> None:
    for _ in range(10):
        print()
    print(os.getcwd())
    print(out_dir)
    
    # WANDBの初期化
    wandb.init(
        project="ResNet18_DANN_domain_adaptation",
        name=f"dann_fold{i_fold}" if i_fold is not None else "dann_holdout",
        config=kargs,
        dir=out_dir,
        tags=["resnet18", "dann", "domain_adaptation"],
    )

    print('device : ', device)
    print('fold   : ', i_fold)
    device = torch.device(device)
    torch.cuda.set_device(device)
    pprint.pprint(kargs, width=100, compact=True, sort_dicts=False)
    print()

    torch.autograd.set_detect_anomaly(True)

    # シード設定
    if kargs['train']['seed'] is not None:
        seed = kargs['train']['seed']
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        g = torch.Generator()
        g.manual_seed(seed)
    else:
        g = None

    # 1. データセット読み込み（Source Domain）
    loader_src, loader_eval_tr, loader_eval_vl = \
        dataset.get_dataset(i_fold=i_fold, generator=g, shuffle=True, **kargs['dataset'])
    
    # 2. ターゲットドメインの読み込み
    if 'dataset_target' in kargs:
        loader_target, _, _ = dataset.get_dataset(
            i_fold=i_fold, generator=g, shuffle=True, **kargs['dataset_target']
        )
        print("✓ Target domain dataset loaded successfully")
    else:
        # ターゲットデータが指定されていない場合はソースと同じものを使用
        print("Warning: 'dataset_target' not found. Using source dataset as target.")
        loader_target = loader_src

    # tllibのForeverIteratorでターゲットデータを無限ループ
    iter_target = ForeverDataIterator(loader_target)

    # 3. バックボーンモデルの構築
    backbone = models.get_model(**kargs['model']).to(device)
    
    # バックボーンの特徴量次元を取得
    if hasattr(backbone, 'fc'):
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()  # 分類層を無効化
    else:
        # ResNet以外のモデルの場合
        num_features = kargs['model'].get('middle', 512)
    
    # バックボーンに特徴量次元を設定
    backbone.out_features = num_features
    
    num_classes = kargs['model'].get('n_class', 2)
    bottleneck_dim = kargs.get('dann', {}).get('bottleneck_dim', 256)
    
    # 4. DANN用のImageClassifierでラップ
    classifier = ImageClassifier(
        backbone=backbone, 
        num_classes=num_classes, 
        bottleneck_dim=bottleneck_dim
    ).to(device)

    # 5. Domain Discriminator
    domain_hidden_size = kargs.get('dann', {}).get('domain_hidden_size', 1024)
    domain_discri = DomainDiscriminator(
        in_feature=classifier.features_dim, 
        hidden_size=domain_hidden_size
    ).to(device)

    # 6. オプティマイザーの設定
    # 分類器とドメイン識別器の両方のパラメータを含める
    all_params = list(classifier.parameters()) + list(domain_discri.parameters())
    
    optimizer = init_opt(all_params, **kargs['opt'])
    
    # 7. 学習率スケジューラ
    if 'scheduler' in kargs:
        if kargs['scheduler']['name'] == 'lambda':
            lr_scheduler = optim.lr_scheduler.LambdaLR(
                optimizer, 
                lambda x: kargs['scheduler'].get('gamma', 0.1) * (1. + kargs['scheduler'].get('decay_rate', 0.001) * float(x)) ** (-0.75)
            )
        else:
            lr_scheduler = None
    else:
        # デフォルトのスケジューラ
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x: 0.01 * (1. + 0.001*float(x)) ** (-0.75))

    # 8. 損失関数の設定
    domain_adv = DomainAdversarialLoss(domain_discri).to(device)
    cls_criterion = nn.CrossEntropyLoss()

    # 既存の処理設定を取得
    pre, post, func, met = models.get_process(device=device, **kargs['process'])

    # 9. カスタム学習ステップの定義
    def train_step(engine, batch):
        classifier.train()
        domain_adv.train()
        optimizer.zero_grad()

        # ソースドメインのバッチ（Ignite Engineから提供）
        x_s, y_s = batch
        x_s, y_s = x_s.to(device), y_s.to(device)

        # 前処理の適用
        if pre is not None:
            x_s = pre(x_s)

        # ターゲットドメインのバッチ（手動で取得）
        try:
            x_t, _ = next(iter_target)
            x_t = x_t.to(device)
            if pre is not None:
                x_t = pre(x_t)
        except StopIteration:
            # イテレータがリセットされた場合
            x_t, _ = next(iter_target)
            x_t = x_t.to(device)
            if pre is not None:
                x_t = pre(x_t)

        # ソースとターゲットを結合
        x = torch.cat((x_s, x_t), dim=0)
        
        # 分類器で予測（logits, features）を取得
        y_pred, f = classifier(x)

        # ソースとターゲットに分割
        y_s_pred, _ = y_pred.chunk(2, dim=0)
        f_s, f_t = f.chunk(2, dim=0)

        # 分類損失（ソースドメインのみ）
        cls_loss = cls_criterion(y_s_pred, y_s)

        # ドメイン敵対損失
        transfer_loss = domain_adv(f_s, f_t)

        # 全体の損失
        trade_off = kargs.get('dann', {}).get('trade_off', 1.0)
        total_loss = cls_loss + trade_off * transfer_loss

        # バックプロパゲーション
        total_loss.backward()
        optimizer.step()
        
        # 学習率スケジューラのステップ
        if lr_scheduler is not None:
            lr_scheduler.step()

        return {
            "loss": total_loss.item(),
            "cls_loss": cls_loss.item(),
            "transfer_loss": transfer_loss.item(),
            "domain_acc": domain_adv.domain_discriminator_accuracy if hasattr(domain_adv, 'domain_discriminator_accuracy') else 0.0
        }

    # 10. 学習エンジンの作成
    trainer = Engine(train_step)

    # 11. 評価用エンジンの作成
    def eval_output_transform(output):
        y_pred, y = output
        # ImageClassifierは (logits, features) のタプルを返すため、logitsのみを取得
        if isinstance(y_pred, tuple):
            y_pred = y_pred[0]
        
        # 後処理の適用
        if post is not None:
            y_pred = post(y_pred)
            
        return y_pred, y

    eval_tr = create_supervised_evaluator(
        classifier, metrics=met, device=device, output_transform=eval_output_transform
    )
    eval_vl = create_supervised_evaluator(
        classifier, metrics=met, device=device, output_transform=eval_output_transform
    )

    # 12. プログレスバーの設定
    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda x: {'loss': x['loss']})

    evals = [eval_tr, eval_vl]
    eval_loaders = [loader_eval_tr, loader_eval_vl]

    # 13. エポック開始時の評価実行
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_and_evaluate(engine):
        # 学習メトリクスのログ
        out = engine.state.output
        log_dict = {
            "epoch": engine.state.epoch,
            "train/loss": out['loss'],
            "train/cls_loss": out['cls_loss'],
            "train/transfer_loss": out['transfer_loss'],
            "train/domain_acc": out['domain_acc'],
        }
        
        if lr_scheduler is not None:
            log_dict["lr"] = optimizer.param_groups[0]['lr']
            
        wandb.log(log_dict)
        
        print(f"Epoch {engine.state.epoch:3d} - Loss: {out['loss']:.4f} "
              f"(Cls: {out['cls_loss']:.4f}, Trans: {out['transfer_loss']:.4f}, "
              f"DomainAcc: {out['domain_acc']:.4f})")

        # 評価の実行
        classifier.eval()
        for mode, ev, loader in zip(["train", "val"], evals, eval_loaders):
            ev.run(loader)
            metrics = ev.state.metrics
            log_dict = {}
            for k, v in metrics.items():
                log_dict[f"{mode}/{k}"] = v
            wandb.log(log_dict)
            print(f"{mode:5s} Metrics: {metrics}")
        classifier.train()

    # 14. TensorBoard Logger
    tb_logger = TensorboardLogger(log_dir=out_dir)
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="train",
        output_transform=lambda x: {
            "loss": x['loss'],
            "cls_loss": x['cls_loss'], 
            "transfer_loss": x['transfer_loss'],
            "domain_acc": x['domain_acc']
        },
    )

    # 15. モデルチェックポイント
    def score_function(engine):
        return engine.state.metrics.get("AUC", 0.0)

    model_checkpoint = ModelCheckpoint(
        out_dir,
        n_saved=1,
        filename_prefix="best",
        score_function=score_function,
        score_name="AUC",
        global_step_transform=global_step_from_engine(trainer),
        require_empty=False
    )

    eval_vl.add_event_handler(Events.COMPLETED, model_checkpoint, {"classifier": classifier})

    # 16. Early Stopping
    if 'stop_ratio' in kargs['train']:
        stopper = early_stopping.EarlyStopping(
            [eval_vl],
            'AUC',
            kargs['train']['stop_ratio'],
            kargs['train']['min_epoch']
        )
        trainer.add_event_handler(Events.EPOCH_COMPLETED, stopper)

    # 17. WandB model watching
    wandb.watch(classifier, log="all", log_freq=50)

    # 18. 学習実行
    try:
        trainer.run(loader_src, max_epochs=kargs['train']['epoch'])
        print("✓ Training completed successfully!")
    except Exception as e:
        print(f"✗ An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        tb_logger.close()
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, type=str)
    parser.add_argument('--fold', '-f', type=int, default=None, help='Fold number (None for holdout)')
    parser.add_argument('--device', '-d', required=True, type=str)
    args = parser.parse_args()

    # 出力ディレクトリの作成
    config_name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = f'../logs/{config_name}_dann'
    
    if args.fold is not None:
        out_dir = f"{out_dir}_fold{args.fold}"
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    print(f"Output Directory: {out_dir}")
    
    # 設定ファイルの読み込み
    config = utils.load_json(args.config)
    utils.save_json(os.path.join(out_dir, 'config.json'), config)
    utils.command_log(out_dir)

    main(args.fold, args.device, out_dir, **config)
    
    utils.save_text(os.path.join(out_dir, 'finish.txt'), '')

