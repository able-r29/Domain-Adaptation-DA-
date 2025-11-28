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

matplotlib.use('Agg')



def init_opt(model:Module, name:str, **karg):
    opt = {
        'sgd'     : optim.SGD,
        'rmsprop' : optim.RMSprop,
        'adam'    : optim.Adam
    }[name]

    dst = opt(model.parameters(), **karg)

    return dst


def main(i_fold:int, device:str, out_dir:str, **kargs) -> None:
    for _ in range(10):
        print()
    print(os.getcwd())
    print(out_dir)
    if os.path.exists(os.path.join(out_dir, 'log')):
        ans = input(out_dir + ' already has log fine. continue? ')
        if ans not in ['y', 'yes', 'Y', 'Yes']:
            print('training canceled')
            exit(0)

    #wandbの初期化
    wandb.init(
        project="ResNet18_val0.2",
        name=f"validation_ratio0.2",
        config=kargs,
        dir=out_dir,
        tags=["resnet18","triplet"] if "mtp" in kargs.get('model', {}).get('name', '') else ["resnet18"],
    )

    print('device : ', device)
    print('fold   : ', i_fold)
    device = torch.device(device)
    torch.cuda.set_device(device)
    # print(json.dumps(kargs, indent=4)) # 長い配列が改行されて見づらいので変更した
    pprint.pprint(kargs, width=100, compact=True, sort_dicts=False)
    print()

    torch.autograd.set_detect_anomaly(True)

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

    loader_train, loader_eval_tr, loader_eval_vl = \
             dataset.get_dataset(i_fold=i_fold, generator=g, shuffle=True, **kargs['dataset'])

    model = models.get_model(**kargs['model']).to(device)
    opt   = init_opt(model, **kargs['opt'])

    pre, post, func, met = models.get_process(device=device, **kargs['process'])

    # デバッグ用：postの内容を確認
    print(f"output_transform type: {type(post)}")
    print(f"output_transform: {post}")

    print(f"metrics type: {type(met)}")
    print(f"metrics keys: {list(met.keys())}")

    trainer = create_supervised_trainer(model,
                                        optimizer       =opt,
                                        loss_fn         =func,
                                        device          =device,
                                        non_blocking    =True,
                                        prepare_batch   =pre,
                                        output_transform=post)
    eval_tr = create_supervised_evaluator(model,
                                          metrics         =met,
                                          device          =device,
                                          non_blocking    =True,
                                          prepare_batch   =pre,
                                          output_transform=post)
    eval_vl = create_supervised_evaluator(model,
                                          metrics         =met,
                                          device          =device,
                                          non_blocking    =True,
                                          prepare_batch   =pre,
                                          output_transform=post)

    evals        = [eval_tr, eval_vl]
    eval_names   = ['eval_tr', 'eval_vl']
    eval_loaders = [loader_eval_tr, loader_eval_vl]

    @trainer.on(Events.EPOCH_STARTED)
    def evaluate(engine):
        model.eval()
        for ev, loader in zip(evals, eval_loaders):
            ev.run(loader)
        model.train()

    pbar = ProgressBar()
    pbar.attach(trainer)
    @trainer.on(Events.EPOCH_STARTED)
    def print_epoch(engine):
        print(f'Epoch : {engine.state.epoch} / {engine.state.max_epochs}')
    
    # wandb ログハンドラー
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(engine):
        wandb.log({
            "epoch": engine.state.epoch,
            "train/loss": engine.state.output
        })

    @eval_tr.on(Events.COMPLETED)
    def log_training_metrics(engine):
        metrics = engine.state.metrics
        log_dict = {"epoch": trainer.state.epoch}
        for metric_name, metric_value in metrics.items():
            log_dict[f"train/{metric_name}"] = metric_value
        wandb.log(log_dict)

    @eval_vl.on(Events.COMPLETED)
    def log_validation_metrics(engine):
        metrics = engine.state.metrics
        log_dict = {"epoch": trainer.state.epoch}
        for metric_name, metric_value in metrics.items():
            log_dict[f"validation/{metric_name}"] = metric_value
        wandb.log(log_dict)

    tb_logger = TensorboardLogger(log_dir=out_dir)

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag       ="train",
        output_transform=lambda loss: {"loss": loss},
        metric_names="all",
    )

    for name, ev in zip(eval_names, evals):
        pbar = ProgressBar()
        pbar.attach(ev)
        tb_logger.attach_output_handler(
            ev,
            event_name=Events.EPOCH_COMPLETED,
            tag=name,
            metric_names=list(met.keys()),
            global_step_transform=global_step_from_engine(trainer),
        )

    def score_function(engine):
        return engine.state.metrics["AUC"]

    model_checkpoint = ModelCheckpoint(
        out_dir,
        n_saved=1,
        filename_prefix="val1",
        score_function=score_function,
        score_name="Accuracy",
        global_step_transform=global_step_from_engine(trainer),
        # require_empty=False, #すでに結果が保存されていた場合に上書きされるようにする
    )
    eval_vl.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

    stopper = early_stopping.EarlyStopping([eval_vl], 'AUC',
                                            kargs['train']['stop_ratio'],
                                            kargs['train']['min_epoch'])
    trainer.add_event_handler(Events.EPOCH_COMPLETED, stopper)

    #モデルをwandbに保存
    wandb.watch(model, log="all", log_freq=30)

    trainer.run(loader_train, max_epochs=kargs['train']['epoch'])

    tb_logger.close()
    wandb.finish()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, type=str)
    parser.add_argument('--fold', '-f', type=int, default=None, help='Fold number (None for holdout)')
    parser.add_argument('--device', '-d', required=True, type=str)
    args = parser.parse_args()

    # torch.multiprocessing.set_start_method('spawn')

    out_dir = '../resnet18_0.2_gamma4'
    print(os.path.join(out_dir, 'finish.txt'))
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    config = utils.load_json(args.config)
    utils.save_json(os.path.join(out_dir, 'config.json'), config)
    print(out_dir)
    utils.command_log(out_dir)

    main(args.fold, args.device, out_dir, **config)
    utils.save_text(os.path.join(out_dir, 'finish.txt'), '')
