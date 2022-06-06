import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import torch
import argparse
import torch.nn as nn
from accelerate import notebook_launcher

from models import LeNet
from loader import Dataloader
from metrics import accuracy
from torchx import Trainer, seed_all, save_state, save_model


def main(*args):
    args = args[0]
    seed_all(args.seed)

    dataloader = Dataloader(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=[False, False],
        pin_memory=True
    )

    model = LeNet(1, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=dataloader.steps_per_epoch, T_mult=1)

    trainer = Trainer(
        model=model,
        train_dataloader=dataloader.train_dataloader,
        val_dataloader=dataloader.val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        config=args.__dict__,
        metrics={'acc': accuracy},
        callbacks={'on_checkpoint': save_state}
    )

    trainer.fit()


if __name__ == '__main__':
    root = Path(__file__).parent

    parser = argparse.ArgumentParser('Training Example')
    parser.add_argument('--data_dir', type=str, default=root/'data')
    parser.add_argument('--log_dir', type=str, default=root/'records')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_processes', type=int, default=1)
    args = parser.parse_args()

    notebook_launcher(main, (args,), num_processes=args.num_processes)
