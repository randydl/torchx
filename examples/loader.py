import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST
from addict import Dict
from torchx import BalancedSampler, DatasetFromSampler


__all__ = ['Dataloader']


class Dataloader:
    def __init__(self, root, **kwargs):
        self.root = root
        self.kwargs = Dict({
            stage: {
                k: v[i] if isinstance(v, (list, tuple)) else v
                for k, v in kwargs.items()
            } for i, stage in enumerate(['train', 'val'])
        })

    @property
    def train_dataset(self):
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((32, 32)),
            T.RandomHorizontalFlip(),
            T.Normalize((0.1307,), (0.3081,))
        ])
        dataset = MNIST(self.root, train=True, download=True, transform=transform)
        sampler = BalancedSampler(dataset.targets, sampling_mode='over')
        dataset = DatasetFromSampler(dataset, sampler)
        return dataset

    @property
    def val_dataset(self):
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((32, 32)),
            T.Normalize((0.1307,), (0.3081,))
        ])
        return MNIST(self.root, train=False, download=True, transform=transform)

    @property
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, **self.kwargs.train)

    @property
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, **self.kwargs.val)

    @property
    def datasets(self):
        return self.train_dataset, self.val_dataset

    @property
    def dataloaders(self):
        return self.train_dataloader, self.val_dataloader

    @property
    def steps_per_epoch(self):
        return len(self.train_dataloader)


if __name__ == '__main__':
    dataloader = Dataloader(
        root = Path(__file__).parent/'data',
        batch_size = 64,
        num_workers = 2,
        shuffle = [False, False],
        pin_memory = True
    )

    print(dataloader.dataloaders)
    print(dataloader.kwargs)
