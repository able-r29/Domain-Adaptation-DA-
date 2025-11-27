import json
import os

from ignite.engine import Engine


def save_json(path, obj, indent=4):
    with open(path, 'w', encoding='utf_8_sig') as f:
        json.dump(obj, f, indent=indent)


def load_json(path):
    with open(path, 'r', encoding='utf-8_sig') as f:
        dst = json.load(f)
    return dst


class LogWriter:
    def __init__(self, out_dir, indent=4, **karg):
        self.save_path = os.path.join(out_dir, 'log')
        self.indent    = indent
        self.items     = karg

        self.values    = load_json(self.save_path) \
                         if os.path.exists(self.save_path) else []
    

    def __call__(self, engine:Engine):
        epoch     = engine.state.epoch
        iteration = engine.state.iteration

        values = {
            'epoch'    :epoch,
            'iteration':iteration,
        }
        for tag, eg in self.items.items():
            for key, val in eg.state.metrics.items():
                name = '/'.join([tag, key])
                values[name] = val
        
        self.values.append(values)

        save_json(self.save_path, self.values, self.indent)
