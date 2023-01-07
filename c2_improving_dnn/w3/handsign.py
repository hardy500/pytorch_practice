import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import Accuracy
from torch.optim.lr_scheduler import StepLR
from handsign_utils import load_dataset
import numpy as np

# GOAL: Handsign recognition

#### Define model ####

class HiddenLayer(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()

    self.hidden = nn.Sequential(
      nn.Linear(in_features, out_features, bias=True),
      nn.BatchNorm1d(hidden_units),
      nn.ReLU(),
    )

  def forward(self, x):
    x = self.hidden(x)
    return x

class Net(nn.Module):
  def __init__(self, in_features, out_features, hidden_units, n_layer):
    super().__init__()

    self.input_layer = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features, hidden_units, bias=True),
      nn.BatchNorm1d(hidden_units),
      nn.ReLU(),
    )

    self.hidden_layer = nn.Sequential(*[HiddenLayer(in_features=hidden_units, out_features=hidden_units) for _ in range(n_layer)])

    self.out_layer = nn.Sequential(
      nn.Linear(hidden_units, out_features, bias=True),
      #nn.BatchNorm1d(out_features)
    )

  def forward(self, x):
    x = nn.functional.normalize(x)
    x = self.input_layer(x)
    x = self.hidden_layer(x)
    x = self.out_layer(x)
    return x

def test_step(model, test_loader, loss_fn, acc_fn):
  test_loss, test_acc  = 0, 0
  model.eval()
  with torch.inference_mode():
    for x, y in test_loader:
      model.train()
      logits      = model(x)
      pred        = torch.softmax(logits, dim=1)
      loss        = loss_fn(logits, y.squeeze().long())
      test_loss += loss
      test_acc  += acc_fn(pred.argmax(1), y.squeeze())

  test_loss = test_loss / len(test_loader)
  test_acc  = test_acc / len(test_loader)
  return test_loss, test_acc

#### Define training and testing steps ####

def train_step(model, train_loader, optimizer, loss_fn, acc_fn):
  train_loss, train_acc = 0, 0
  for x, y in train_loader:
    model.train()
    logits      = model(x)
    pred        = torch.softmax(logits, dim=1)
    loss        = loss_fn(logits, y.squeeze().long()) # NOTE: CrossEntropyLoss has auto softmax, pass in logits is enought
    train_loss += loss
    train_acc  += acc_fn(pred.argmax(1), y.squeeze())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  train_loss /= len(train_loader)
  train_acc  /= len(train_loader)
  return train_loss, train_acc

def test_dev_split(test_set, n_folds):
    test_size = int((n_folds-1) / n_folds * len(test_set))
    dev_size  = len(test_set) - test_size
    test_dataset, dev_dataset = torch.utils.data.random_split(test_set, [test_size, dev_size])
    return test_dataset, dev_dataset

def train(model, test_dataloader, training_set, optimizer, loss_fn, acc_fn, epochs, n_folds, lr):
  for i in range(n_folds):
    # Split data into test and dev sets for this fold
    train_dataset, dev_dataset = test_dev_split(training_set, n_folds)

    # Create data loader for train and dev sets
    train_dataloader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
    dev_dataloader    = DataLoader(dev_dataset, batch_size=64, shuffle=False)

    print(f"lr: {lr}")
    for epoch in range(epochs+1):
      train_loss, train_acc = train_step(model, train_dataloader, optimizer, loss_fn, acc_fn)
      dev_loss , dev_acc    = test_step(model, dev_dataloader, loss_fn, acc_fn)
      test_loss, test_acc   = test_step(model, test_dataloader, loss_fn, acc_fn)

      if epoch % 10 == 0:
        print(f"[{epoch}/{epochs}] Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f}\
               || Dev loss: {dev_loss:.3f} | Dev acc: {dev_acc:.3f}\
               || Test loss: {test_loss:.3f} | Dev acc: {test_acc:.3f}")
    print()

#### Preprocessing data ####

# size (1080, 64, 64, 3), (1, 1080)
x_train_og, y_train_og, x_test_og, y_test_og , classes = load_dataset()

x_train      = torch.from_numpy(x_train_og).float()
y_train      = torch.from_numpy(y_train_og.T).float()
train_set    = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

x_test       = torch.from_numpy(x_test_og).float()
y_test       = torch.from_numpy(y_test_og.T).float()
test_set     = TensorDataset(x_test, y_test)
test_loader  = DataLoader(test_set, batch_size=64)

#### Training ####

import random

in_features  = 64*64*3
out_features = 6
hidden_units = 5
n_layer      = 2

lambd        = 0.001   # hyperparameter for l2 reg
k_folds      = 5       # k-fold cross-validation
momentum     = 0.99

model     = Net(in_features, out_features, hidden_units, n_layer)
loss_fn   = nn.CrossEntropyLoss()
acc_fn    = Accuracy(task='multiclass', num_classes=out_features)
#optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambd)

# torch.manual_seed(0)
# Tryout random learning rate
for i in range(5):
  r = -4*np.random.rand()
  lr = 10**r
  optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=lambd)
  epochs       = 30
  train(model, test_loader, train_set, optimizer, loss_fn, acc_fn, epochs, k_folds, lr)
