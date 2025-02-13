import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

# Dummy dataset class
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.data = torch.randn(length, size)
        self.labels = torch.randint(0, 2, (length,))
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)

# Simple model
class SimpleModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# Training function
def train(rank, world_size, epochs=5, batch_size=4096, input_size=100):
    torch.manual_seed(42)
    
    # Initialize process group for DDP
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = SimpleModel(input_size, num_classes=2).to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scaler = amp.GradScaler()
    criterion = nn.CrossEntropyLoss()
    dataset = RandomDataset(input_size, 1000)
    
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if world_size > 1 else None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler)
    
    model.train()
    for epoch in range(epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            with amp.autocast():  # Enable mixed precision
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        if rank == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size > 1:
        torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    else:
        train(rank=0, world_size=1)
