from ignite.engine import Events, Engine



class LogBest:
    def __init__(self, key, ratio):
        self.key      = key
        self.val_best = None
        self.ep_best  = 0
        self.ep_total = 0
        self.ratio    = ratio


    def __call__(self, engine):
        self.ep_total += 1

        value = engine.state.metrics[self.key]
        if self.val_best is None or value > self.val_best:
            self.val_best = value
            self.ep_best  = self.ep_total


    def is_saturate(self):
        return self.ep_total / self.ep_best > self.ratio



class EarlyStopping:
    def __init__(self, evals, key, stop_ratio, min_epoch=10) -> None:
        self.loggers   = []
        self.min_epoch = min_epoch
        for ev in evals:
            logger = LogBest(key, stop_ratio)
            ev.add_event_handler(handler=logger, event_name=Events.COMPLETED)
            self.loggers.append(logger)

    def __call__(self, engine:Engine):
        if engine.state.epoch > self.min_epoch and \
           all(l.is_saturate() for l in self.loggers):
            engine.terminate()
    

    def best_average(self):
        return sum(l.val_best for l in self.loggers) / len(self.loggers)
