import random
from typing import Tuple
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
import torch.utils.data as data

from logger import Logger
from sampler import ContinualSampler
logger = Logger()
def get_task_dataset(task_id: int) -> Tuple[data.Dataset, data.Dataset]:
    """
    Returns train and validation datasets for a given task
    For CIFAR-10, each task consists of 2 classes
    """
    train_dataset = torchvision.datasets.CIFAR10(
        "data",
        train=True,
        download=True,
        transform=Compose([ToTensor(), Normalize(0.5, 0.5)]),
    )
    val_dataset = torchvision.datasets.CIFAR10(
        "data",
        train=False,
        download=True,
        transform=Compose([ToTensor(), Normalize(0.5, 0.5)]),
    )

    # Calculate class range for this task (2 classes per task)
    start_class = task_id * 2
    end_class = start_class + 2

    # Filter for classes in the current task
    train_mask = torch.logical_and(
        torch.tensor(train_dataset.targets) >= start_class,
        torch.tensor(train_dataset.targets) < end_class,
    )
    val_mask = torch.logical_and(
        torch.tensor(val_dataset.targets) >= start_class,
        torch.tensor(val_dataset.targets) < end_class,
    )

    train_dataset = torch.utils.data.Subset(train_dataset, torch.where(train_mask)[0])

    # Preload datasets to GPU memory
    train_data = torch.stack([x[0] for x in train_dataset]).cuda()
    train_targets = torch.tensor([x[1] for x in train_dataset]).cuda()
    train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
    
    
    val_dataset = torch.utils.data.Subset(val_dataset, torch.where(val_mask)[0])
    val_data = torch.stack([x[0] for x in val_dataset]).cuda()
    val_targets = torch.tensor([x[1] for x in val_dataset]).cuda()
    val_dataset = torch.utils.data.TensorDataset(val_data, val_targets)

    return train_dataset, val_dataset


def reduce_dataset(dataset, reduction_factor):
    size = len(dataset)
    reduced_size = int(size * reduction_factor)
    indices = random.sample(range(size), reduced_size)
    return data.Subset(dataset, indices)


def get_pretrain_dataset(task_id,args):
    
    loaded_datasets = []
    
    for i in range(task_id +1):
        train_dataset, val_dataset = get_task_dataset(i)
        
        if i == task_id:
            loaded_datasets.append(train_dataset)
        else:
            if args.reduction_factor == 1:
                loaded_datasets.append(train_dataset)
            elif args.reduction_factor == 0:
                pass
            else:
                loaded_datasets.append(reduce_dataset(train_dataset, args.reduction_factor))
    
    if len(loaded_datasets) == 1:
        return loaded_datasets[0]
    else:
        ds = data.ConcatDataset(loaded_datasets)
        logger.print(f"Pretrain dataset size: {len(ds)}")
        sizes = [len(d) for d in ds.datasets]
        logger.print(f"Individual Sizes: {sizes}")
        return ds


def get_finetune_dataset(task_id,args):
    loaded_datasets = []
    loaded_val_datasets = []
    for i in range(task_id + 1):
        train_dataset, val_dataset = get_task_dataset(i)

        loaded_datasets.append(train_dataset)
        loaded_val_datasets.append(val_dataset)

    if len(loaded_datasets) == 1:
        return loaded_datasets[0], loaded_val_datasets[0]
    else:
        ds = data.ConcatDataset(loaded_datasets)
        val_ds = data.ConcatDataset(loaded_val_datasets)
        logger.print(f"Finetune dataset size: {len(ds)}")
        sizes = [len(d) for d in ds.datasets]
        logger.print(f"Individual Sizes: {sizes}")

    return ds, val_ds


def get_pretrain_dataloader(task_id,args):
    load_batch_size = min(args.max_device_batch_size, args.batch_size)
    assert args.batch_size % load_batch_size == 0
    
    pretrain_dataset = get_pretrain_dataset(task_id,args)
    
    if isinstance(pretrain_dataset,data.ConcatDataset):
        sampler = ContinualSampler(pretrain_dataset)
    else:
        sampler = data.RandomSampler(pretrain_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        pretrain_dataset, load_batch_size, sampler=sampler
    )


    return train_dataloader

def get_finetune_dataloader(task_id,args):
    finetune_batch_size = args.finetune_batch_size
    
    finetune_dataset, val_dataset = get_finetune_dataset(task_id,args)
    
    train_dataloader = torch.utils.data.DataLoader(
        finetune_dataset, finetune_batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, finetune_batch_size * 8, shuffle=False
    )
    
    return train_dataloader, val_dataloader