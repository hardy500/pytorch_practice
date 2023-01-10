import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn


### MNIST DATASET ###
BS            = 128
train_dataset = datasets.MNIST(root='data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='data',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BS,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BS,
                         shuffle=False)

# Checking the dataset
# torch.Size([128, 1, 28, 28]) torch.Size([128])
#for x, y in train_loader:
#  print(x.shape, y.shape)
#  break

### Model ###

class ResidualBlock(nn.Module):
  def __init__(self, channels):
    super().__init__()

    self.block = nn.Sequential(
      nn.Conv2d(in_channels=channels[0],
                out_channels=channels[1],
                kernel_size=3,
                stride=2,
                padding=1),
      nn.BatchNorm2d(channels[1]),

      nn.ReLU(inplace=True),

      nn.Conv2d(in_channels=channels[1],
                out_channels=channels[2],
                kernel_size=1,
                stride=1,
                padding=0),
      nn.BatchNorm2d(channels[2])
    )

    self.shortcut = nn.Sequential(
      nn.Conv2d(in_channels=channels[0],
                out_channels=channels[2],
                kernel_size=1,
                stride=2,
                padding=0),
      nn.BatchNorm2d(channels[2])
    )

  def forward(self, x):
    shortcut = x
    block    = self.block(x)
    shortcut = self.shortcut(x)
    x        = nn.functional.relu(block + shortcut)
    return x

class ConvNet2(nn.Module):
  def __init__(self, n_classes):
    super().__init__()

    self.res_block_1 = ResidualBlock(channels=[1, 4, 8])
    self.res_block_2 = ResidualBlock(channels=[8, 16, 32])

    self.linear_1    = nn.Linear(7*7*32, n_classes)

  def forward(self, x):
    out    = self.res_block_1(x)
    out    = self.res_block_2(out)
    logits = self.linear_1(out.view(-1, 7*7*32))
    return logits

class IdBlock(nn.Module):
  def __init__(self, n_classes):
    super().__init__()

    # First residual block
    self.block_1 = nn.Sequential(
        nn.Conv2d(in_channels=1,
                  out_channels=4,
                  kernel_size=1,
                  stride=1,
                  padding=0),

        nn.BatchNorm2d(4),
        nn.ReLU(),

        nn.Conv2d(in_channels=4,
                  out_channels=1,
                  kernel_size=3,
                  stride=1,
                  padding=1),

        nn.BatchNorm2d(1)
    )

    self.block_2 = nn.Sequential(
        nn.Conv2d(in_channels=1,
                  out_channels=4,
                  kernel_size=1,
                  stride=1,
                  padding=0),

        nn.BatchNorm2d(4),
        nn.ReLU(),

        nn.Conv2d(in_channels=4,
                  out_channels=1,
                  kernel_size=3,
                  stride=1,
                  padding=1),

        nn.BatchNorm2d(1)
    )

    # Fully connected
    self.linear_1 = nn.Linear(1*28*28, n_classes)

  def forward(self, x):
    # First residual block
    shortcut = x
    x = self.block_1(x)
    x = nn.functional.relu(x + shortcut)

    # Second residual block
    shortcut = x
    x = self.block_2(x)
    x = nn.functional.relu(x + shortcut)

    # Fully connect
    logits = self.linear_1(x.view(-1, 1*28*28))
    return logits

def accuracy(model, data_loader):
  correct_pred, num_examples = 0, 0
  for i, (x, y) in enumerate(data_loader):
    x, y = x.to(DEVICE), y.to(DEVICE)

    logits = model(x)
    _, pred = torch.max(logits, 1)
    num_examples += y.size(0)
    correct_pred += (pred == y).sum()

  return correct_pred.float()/num_examples


### Settings ###
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
RANDOM_SEED = 0
LR          = 0.01
EPOCHS      = 1

# Architecture
N_CLASSES = 10

torch.manual_seed(RANDOM_SEED)
model = ConvNet2(n_classes=N_CLASSES)
model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

start_time = time.time()
for epoch in range(EPOCHS):
  model = model.train()
  for batch_idx, (x, y) in enumerate(train_loader):
    x, y = x.to(DEVICE), y.to(DEVICE)

    logits = model(x)
    cost = nn.functional.cross_entropy(logits, y)
    optimizer.zero_grad()

    cost.backward()
    optimizer.step()

  model = model.eval()
  with torch.inference_mode():
    print('Epoch: %03d/%03d training accuracy: %.2f%%' % (epoch+1, EPOCHS, accuracy(model, train_loader)))

  print("Time elapsed: %.2f min" % ((time.time() - start_time)/60))

print("Total Trainint Time: %2.f min" % ((time.time() - start_time)/60))
print("Test accuracy: %.2f" % (accuracy(model, test_loader)))
