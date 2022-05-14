import torch
from addict import Dict
from pathlib import Path
from accelerate import Accelerator
from torchx.utils import StatsTracker


__all__ = ['Trainer']


callbacks_whitelist = [
    'compute_loss',
    'compute_metrics',
    'on_checkpoint'
]


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, config={}, metrics={}, callbacks={}):
        version = 'version' + str(config.get('version', 0))
        self.log_dir = Path(config.get('log_dir', Path.cwd()/'records'))/version

        self.accelerator = Accelerator(log_with='all', logging_dir=self.log_dir.parent)
        if self.is_main_process: self.init_trackers(version)

        self.model = self.prepare(model)
        self.train_dataloader = self.prepare(train_dataloader)
        self.val_dataloader = self.prepare(val_dataloader)
        self.optimizer = self.prepare(optimizer)
        self.scheduler = self.prepare(scheduler)
        self.criterion = criterion
        self.config = Dict(config)
        self.metrics = metrics

        for k, fn in callbacks.items():
            if k not in callbacks_whitelist: continue
            setattr(Trainer, k, fn)

    def __getattr__(self, x):
        return getattr(self.accelerator, x)

    @property
    def last_lr(self):
        return self.scheduler.get_last_lr()[0]

    def fit(self):
        num_epochs = self.config.num_epochs or 1
        self.print('-'*10, 'Start Training', '-'*10)
        for epoch in range(num_epochs):
            self.print(f'epoch: {epoch}/{num_epochs}')
            train_stats = self.train(epoch)
            val_stats = self.validate(epoch)
            self.on_checkpoint(num_epochs, epoch, train_stats, val_stats)
        self.print('-'*10, 'Finish Training', '-'*10)
        self.end_training()

    def train(self, epoch):
        self.model.train()
        stats = StatsTracker()
        for step, (inputs, targets) in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.compute_loss(outputs, targets)
            metrics = self.compute_metrics(outputs, targets)
            self.backward(loss)
            self.optimizer.step()
            self.scheduler.step()

            stats.update('train_loss', loss, len(inputs))
            stats.update_dict({f'train_{k}': v for k, v in metrics.items()}, len(inputs))
            self.log(dict(stats.avg, **{'lr': self.last_lr}), epoch * len(self.train_dataloader) + step)
        self.print(stats)
        return stats

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        stats = StatsTracker()
        for step, (inputs, targets) in enumerate(self.val_dataloader):
            outputs = self.model(inputs)
            outputs = self.gather(outputs)
            targets = self.gather(targets)
            loss = self.compute_loss(outputs, targets)
            metrics = self.compute_metrics(outputs, targets)

            stats.update('val_loss', loss, len(inputs))
            stats.update_dict({f'val_{k}': v for k, v in metrics.items()}, len(inputs))
            self.log(stats.avg, epoch * len(self.val_dataloader) + step)
        self.print(stats)
        return stats

    def compute_loss(self, *args):
        return self.criterion(*args)

    @torch.no_grad()
    def compute_metrics(self, *args):
        return {k: fn(*args) for k, fn in self.metrics.items()}

    def on_checkpoint(self, *args):
        pass
