import torch
from addict import Dict
from pathlib import Path
from accelerate import Accelerator
from torchx.utils import StatsTracker


__all__ = ['Trainer']


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, metrics={}, config={}):
        self.accelerator = Accelerator(
            log_with = 'all',
            logging_dir = Path(config.get('log_dir', Path.cwd()/'records'))
        )
        if self.is_main_process: self.init_trackers('logs')

        self.model = self.prepare(model)
        self.train_dataloader = self.prepare(train_dataloader)
        self.val_dataloader = self.prepare(val_dataloader)
        self.optimizer = self.prepare(optimizer)
        self.scheduler = self.prepare(scheduler)
        self.criterion = criterion
        self.metrics = metrics
        self.config = Dict(config)

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
            self.save_model(num_epochs, epoch, train_stats, val_stats)
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

    def save_model(self, *args):
        self.wait_for_everyone()
        model = self.unwrap_model(self.model)

        save_dir = self.logging_dir/'states'
        save_dir.mkdir(parents=True, exist_ok=True)

        self.save(model.state_dict(), save_dir/'epoch_{:0>{}d}-tl_{:.4f}-vl_{:.4f}.pth'.format(
            args[1], len(str(args[0] - 1)),
            args[2].train_loss.avg,
            args[3].val_loss.avg
        ))
