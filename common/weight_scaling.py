import torch
from torch.nn import Module
from ignite.engine import Engine, Events


class WeightScaling:
    def __init__(self, model:Module, keys:list, limit:float):
        self.model = model
        self.keys  = keys
        self.limit = limit


    def __call__(self, _):
        for n, p in self.model.named_parameters():
            if all([k in n for k in self.keys]):
                m = torch.max(torch.abs(p)).item()
                if m > self.limit:
                    p.data *= self.limit / m 



def attach(engine:Engine, model:Module, keys:list, limit:float):
    scaler = WeightScaling(model, keys, limit)
    engine.add_event_handler(Events.ITERATION_COMPLETED, scaler)

    return scaler
