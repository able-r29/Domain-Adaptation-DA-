import time

from ignite.engine import Engine, Events


class MetricsPreview:
    def __init__(self, loader, metrics, ema=0.98, tag=None, base_engine:Engine=None):
        self.metrics     = metrics
        self.epoch_itr   = len(loader)
        self.ema         = ema
        self.tag         = tag
        self.base_engine = base_engine

        self.start_time = 0
        self.this_itr   = 0

        for met in ['epoch', 'iteration', 'progress', 'epoch-time'] + metrics:
            print('{:^10s}'.format(met), end='  ')
        if tag:
            print('{:^10s}'.format('tag'), end='  ')
        self.sums   = {}
        self.values = {}
        print()
    

    def start_epoch(self, _):
        self.start_time = time.time()
        self.this_itr   = 0
        self.sums       = {}
    

    def print_values(self, epoch, iteration, progress, past_time, metrics, newline=False):
        print('\r', end='')
        print('{:^10d}'  .format(epoch),     end='  ')
        print('{:^10d}'  .format(iteration), end='  ')
        print('{:^10.3%}'.format(progress),  end='  ')
        print('{:^10d}'  .format(past_time), end='  ')

        for met in self.metrics:
            val = metrics[met]
            print('{:^ .3e}'.format(val), end='  ')
        if self.tag:
            print('{:^10s}'.format(self.tag), end='  ')

        if newline:
            print('', flush=True)
        else:
            print('', flush=True, end='')


    def end_iteration(self, engine:Engine):
        past_time       = int(time.time() - self.start_time)
        self.this_itr  += 1

        epoch     = (self.base_engine if self.base_engine else engine).state.epoch
        iteration = (self.base_engine if self.base_engine else engine).state.iteration
        progress  = self.this_itr / self.epoch_itr

        for met in self.metrics:
            val = engine.state.output[met]
            if self.ema > 0:
                if met not in self.values:
                    self.values[met] = val
                else:
                    self.values[met] = self.values[met] * self.ema + val * (1-self.ema)
            else:
                if met not in self.sums:
                    self.sums[met] = 0
                self.sums[met] += val
                self.values[met] = self.sums[met] / self.this_itr
            
        self.print_values(epoch, iteration, progress, past_time, self.values)
    

    def end_epoch(self, engine:Engine):
        past_time = int(time.time() - self.start_time)

        epoch     = (self.base_engine if self.base_engine else engine).state.epoch
        iteration = (self.base_engine if self.base_engine else engine).state.iteration
        progress  = 1

        self.print_values(epoch, iteration, progress, past_time, self.values, newline=True)



def attach(engine:Engine, loader, metrics, ema=0.98, tag=None, base_engine:Engine=None):
    viewr = MetricsPreview(loader, metrics, ema=ema, tag=tag, base_engine=base_engine)
    engine.add_event_handler(Events.EPOCH_STARTED,       viewr.start_epoch)
    engine.add_event_handler(Events.ITERATION_COMPLETED, viewr.end_iteration)
    engine.add_event_handler(Events.EPOCH_COMPLETED,     viewr.end_epoch)

    return viewr
