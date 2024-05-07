# -*- coding: utf-8 -*-
"""DDP_Trainer.ipynb

## Distributed Training with Multiple GPUs
#### Distrbuted Data Parallel
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data import DataLoader

"""### Declare Model"""

class NN(torch.nn.Module):
  def __init__(self):
    super(NN, self).__init__()
    self.model = torch.nn.Sequential(
        torch.nn.Linear(784, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10),
        torch.nn.LogSoftmax(dim=1)
    )

  def forward(self, x):
    print(f"Model[{torch.cuda.current_device()}] Input: ", x.shape)
    return self.model(x)

def train(model, data, device, criterion, optimizer, epochs=10):
  for epoch in range(epochs):
    total_loss = 0.0
    count = 0.0
    for inputs, labels in data:
      inputs = inputs.to(device)
      labels = labels.to(device)
      output = model(inputs)
      loss = criterion(output, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      count += 1

    print(f"GPU-{torch.cuda.current_device()}| Epoch:{epoch}| Loss: {total_loss / count}")

def eval(model, data, device):
  total, correct = 0, 0
  for inputs, labels in data:
    inputs = inputs.to(device)
    labels = labels.to(device)
    output = model(inputs)
    output = output.argmax(dim=1)
    correct_predictions = (output == labels).sum()
    total_predictions = labels.size()[0]
    total += total_predictions
    correct += correct_predictions
  print(f"GPU-{torch.cuda.current_device()}|Accuracy: {correct / total}")

def main():
  # Initialize process group
  dist.init_process_group(backend='nccl')

  # Setup Device and Model
  local_rank = int(os.environ['LOCAL_RANK'])
  device = torch.device('cuda', local_rank)
  torch.cuda.set_device(device)

  model = NN().to(device)
  model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

  # Setup Optimizers and Loss function
  optimizer = torch.optim.Adam(model.parameters())
  criterion = torch.nn.CrossEntropyLoss()

  # Load Training data
  training_data = datasets.MNIST(
      root="data",
      train=True,
      download=True,
      transform=transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=0.0, std=1.0),
                                    transforms.Lambda(lambda x: torch.flatten(x))])
  )

  test_data = datasets.MNIST(
      root="data",
      train=False,
      download=True,
      transform=transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=0.0, std=1.0),
                                    transforms.Lambda(lambda x: torch.flatten(x))])
  )

  train_sampler = torch.utils.data.distributed.DistributedSampler(training_data)
  test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
  train_dataloader = DataLoader(training_data, batch_size=64, sampler=train_sampler)
  test_dataloader = DataLoader(test_data, batch_size=64, sampler=test_sampler)

  model.train()
  train(model, train_dataloader, device, criterion, optimizer, epochs=10)
  model.eval()
  eval(model, test_dataloader, device)
  if local_rank == 0:
    torch.save(model.state_dict(), 'checkpoint.pt')

if __name__ == "__main__":
  main()