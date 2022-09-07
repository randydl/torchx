import torch
import random
import numpy as np
import pandas as pd
from addict import Dict
from collections import defaultdict


__all__ = [
    'seed_all',
    'AverageMeter',
    'StatsTracker',
    'BalancedSampler'
]


def seed_all(seed=42, deterministic=True, benchmark=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += self.val * int(n)
        self.count += int(n)
        self.avg = self.sum / self.count

    def __repr__(self):
        fn = lambda v: 0 if isinstance(v, int) else 4
        return ', '.join([f'{k}: {v:.{fn(v)}f}' for k, v in self.__dict__.items()])


class StatsTracker:
    def __init__(self, prefix=None, sep='_'):
        self.stats = defaultdict(AverageMeter)
        self.prefix = '' if prefix is None else prefix + sep

    def update(self, key, val, n=1):
        self.stats[self.prefix + key].update(val, n)

    def update_dict(self, dic, n=1):
        for k, v in dic.items(): self.update(k, v, n)

    def __getattr__(self, x):
        if x in self.stats.keys():
            return self.stats[x]
        elif self.prefix + x in self.stats.keys():
            return self.stats[self.prefix + x]
        return Dict({k: getattr(v, x) for k, v in self.stats.items()})

    def __repr__(self):
        return ', '.join([f'{k}: {v:.4f}' for k, v in self.avg.items()])


class BalancedSampler(torch.utils.data.Sampler):
    def __init__(self, inputs, sampling_mode='same', get_labels=None):
        labels = inputs if get_labels is None else get_labels(inputs)
        labels = pd.Series(labels)
        counts = labels.value_counts()

        if sampling_mode == 'down':
            length = len(counts) * counts.min()
        elif sampling_mode == 'over':
            length = len(counts) * counts.max()
        elif sampling_mode == 'same':
            length = len(labels)
        elif isinstance(sampling_mode, int) and sampling_mode > 0:
            length = sampling_mode
        else:
            raise ValueError("sampling_mode must be one of ['down', 'over', 'same'] or a positive integer")

        self.length = length
        self.weights = 1 / counts[labels].values

    def __iter__(self):
        sample = pd.Series(range(len(self.weights))).sample(self.length, weights=self.weights, replace=True)
        return iter(sample.tolist())

    def __len__(self):
        return self.length
