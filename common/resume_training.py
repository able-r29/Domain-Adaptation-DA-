import json
import os
from typing import Dict

import torch
from torch.nn import Module
from ignite.engine import Engine, Events


def load_json(path):
    with open(path, 'r', encoding='utf-8_sig') as f:
        dst = json.load(f)
    return dst


def listup_models(dirname:str):
    files = [os.path.join(dirname, f) for f in os.listdir(dirname)]
    dst   = {}
    for f in files:
        basename = os.path.basename(f)
        _, name, iteration = basename.split('_')
        iteration = int(iteration[:-4])
        
        if name not in dst:
            dst[name] = []
        dst[name].append((f, iteration))

    dst = {name:[f[0] for f in sorted(dst[name], key=lambda x: x[1])] for name in dst}
    return dst


def load_model(dirname:str, to_load:Dict[str, Module]):
    if not os.path.exists(dirname):
        print(dirname, 'is not exists')
        return
        
    files = listup_models(dirname)
    for k, m in to_load.items():
        if k not in files:
            print('loading', k, 'is skipped')
            continue
        path = files[k][-1]
        m.load_state_dict(torch.load(path, map_location='cpu'))
        print(path, 'is loaded')


def resume(engine:Engine, dirname:str, to_load:Dict[str, Module], model_dir='models'):
    if not os.path.exists(dirname):
        print(dirname, 'is not exists')
        return

    model_dir = os.path.join(dirname, model_dir)
    load_model(model_dir, to_load)


    log_name  = os.path.join(dirname, 'log')
    values    = load_json(log_name) if os.path.exists(log_name) else [{'epoch':0, 'iteration':0}]
    epoch     = values[-1]['epoch']
    iteration = values[-1]['iteration']

    def _resume(eg:Engine):
        eg.state.epoch     = epoch
        eg.state.iteration = iteration
        print('training is', epoch, 'epoch, ', iteration, 'iteration')

    engine.add_event_handler(Events.STARTED, _resume)
