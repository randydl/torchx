import torch
import numpy as np
import pandas as pd
from pathlib import Path


__all__ = [
    'save_state',
    'save_model'
]


def save_state(trainer, num_epochs, epoch, train_stats, val_stats):
    trainer.wait_for_everyone()
    fname = 'epoch_{}'.format(epoch)
    trainer.save_state(trainer.log_dir/fname)


def save_model(trainer, num_epochs, epoch, train_stats, val_stats):
    trainer.wait_for_everyone()
    model = trainer.unwrap_model(trainer.model)
    fname = 'epoch_{}-tl_{:.4f}-vl_{:.4f}.pth'.format(
        epoch,
        train_stats.train_loss.avg,
        val_stats.val_loss.avg
    )
    trainer.save(model.state_dict(), trainer.log_dir/fname)
