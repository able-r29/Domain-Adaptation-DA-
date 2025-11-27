import json
import os

import matplotlib.pyplot as plt
from ignite.engine import Engine


def load_json(path):
    with open(path, 'r', encoding='utf-8_sig') as f:
        dst = json.load(f)
    return dst


def load_log(out_dir:str):
    values = {
        'epoch'    :[],
        'iteration':[]
    }
    save_path = os.path.join(out_dir, 'log')
    if os.path.exists(save_path):
        log = load_json(save_path)
        for l in log:
            for k in l:
                if k not in values:
                    values[k] = []
                values[k].append(l[k])

    return values


class LogPlotter:
    def __init__(self, out_dir:str, targets:list, xkey:str='epoch', **karg):
        self.out_dir = out_dir
        self.targets = targets
        self.xkey    = xkey
        self.items   = karg
        
        self.values = load_log(out_dir)


    def __call__(self, engine:Engine):
        epoch     = engine.state.epoch
        iteration = engine.state.iteration
        self.values['epoch']    .append(epoch)
        self.values['iteration'].append(iteration)

        for target in self.targets:
            for tag, eg in self.items.items():
                name = '/'.join([tag, target])
                if name not in self.values:
                    self.values[name] = []
                val = eg.state.metrics[target]
                self.values[name].append(val)
        
        x = self.values[self.xkey]

        for target in self.targets:
            path = os.path.join(self.out_dir, target + '.png')
            axs = plt.axes()
            for tag, _ in self.items.items():
                name = '/'.join([tag, target])
                y = self.values[name]
                plt.plot(x, y, label=name)
            plt.legend()
            plt.xlabel(self.xkey)
            axs.yaxis.grid()
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
