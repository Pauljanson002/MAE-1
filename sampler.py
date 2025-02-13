import torch
from torch.utils.data import ConcatDataset, Subset
from torchvision import datasets, transforms
import random
import numpy as np
from itertools import cycle
import math
import sys
import os



class ContinualSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        if not isinstance(dataset, ConcatDataset):
            raise ValueError("Dataset must be a ConcatDataset")
        self.dataset = dataset
        self.task_id = len(dataset.datasets) - 1
        self.n_tasks = len(dataset.datasets)

        self.dataset_sizes = [len(d) for d in dataset.datasets]
        self.cumulative_sizes = dataset.cumulative_sizes
        self.task_samples = []
        self.previous_samples = []
        for i in range(self.n_tasks):
            if i == self.task_id:
                self.task_samples.extend(
                    range(
                        self.cumulative_sizes[i - 1] if i > 0 else 0,
                        self.cumulative_sizes[i],
                    )
                )
            elif i < self.task_id:
                self.previous_samples.extend(
                    range(
                        self.cumulative_sizes[i - 1] if i > 0 else 0,
                        self.cumulative_sizes[i],
                    )
                )

        random.shuffle(self.task_samples)
        random.shuffle(self.previous_samples)

    def __iter__(self):
        task_cycle = cycle(self.task_samples)
        previous_cycle = cycle(self.previous_samples)

        for _ in range(len(self)):
            yield next(task_cycle)
            if self.task_id > 0:  # Only yield from previous if there are previous tasks
                yield next(previous_cycle)

    def __len__(self):
        return (
            len(self.task_samples) * 2 if self.task_id > 0 else len(self.task_samples)
        )
