from planar_utils import load_planar_dataset

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import Accuracy

# NOTE: Dataset is not linearly separable
# GOAL: Train a network with hidden layer such that it can fit the data

def load_data():
  x_train_og, y_train_og = load_planar_dataset()
  # shape (400, 2), (400, 1)
  x_train      = torch.from_numpy(x_train_og.T).float()
  y_train      = torch.from_numpy(y_train_og.T).float()
  train_set    = TensorDataset(x_train, y_train)
  train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
  return train_loader

#### Define model ####

class HiddenLayer(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()

    self.hidden = nn.Sequential(
      nn.Linear(in_features, out_features, bias=False),
      nn.ReLU(),
    )

  def forward(self, x):
    x = self.hidden(x)
    return x

class Net(nn.Module):
  def __init__(self, in_features, out_features, hidden_units, n_layer):
    super().__init__()

    self.input_layer = nn.Sequential(
      nn.Linear(in_features, hidden_units, bias=False),
      nn.ReLU(),
    )

    self.hidden_layer = nn.Sequential(*[HiddenLayer(in_features=hidden_units, out_features=hidden_units) for _ in range(n_layer)])

    self.out_layer = nn.Sequential(
      nn.Linear(hidden_units, out_features, bias=False),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.input_layer(x)
    x = self.hidden_layer(x)
    x = self.out_layer(x)
    return x

def train_step(model, train_loader, optimizer, loss_fn, acc_fn):
  train_loss, train_acc = 0, 0
  for x, y in train_loader:
    model.train()

    pred        = model(x)
    loss        = loss_fn(pred, y)
    train_loss += loss
    train_acc  += acc_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  train_loss /= len(train_loader)
  train_acc  /= len(train_loader)
  return train_loss, train_acc

def train(model, train_loader, optimizer, loss_fn, acc_fn, epochs):
  for epoch in range(epochs+1):
    train_loss, train_acc = train_step(model, train_loader, optimizer, loss_fn, acc_fn)

    if epoch % 100 == 0:
      print(f"[{epoch}/{epochs}] Train loss: {train_loss:.4f} | Train acc: {train_acc:.3f}")

if __name__ == "__main__":
  train_loader = load_data()
  in_features  = 2
  out_features = 1
  hidden_units = 4
  n_layer      = 3

  #### Training ####

  torch.manual_seed(1)
  model     = Net(in_features, out_features, hidden_units, n_layer)
  loss_fn   = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
  acc_fn    = Accuracy(task='binary', num_classes=1)

  epochs = 900
  train(model, train_loader, optimizer, loss_fn, acc_fn, epochs)
  # result: [100/100] Train loss: 0.2595 | Train acc: 0.915