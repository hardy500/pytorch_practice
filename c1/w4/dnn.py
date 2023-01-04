import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import Accuracy
from dnn_utils import load_data

# GOAL: Classify cat vs non-cat with dnn
# NOTE: Architecture are pretty such the same as in planar data classification

#### Preprocessing data ####

# size (209, 64, 64, 3) (1, 209)
def load_dataset():
  x_train_og, y_train_og, x_test_og, y_test_og, classes = load_data()

  x_train      = torch.from_numpy(x_train_og).float()
  y_train      = torch.from_numpy(y_train_og.T).float()
  train_set    = TensorDataset(x_train, y_train)
  train_loader = DataLoader(train_set, batch_size=32, shuffle=True)

  x_test       = torch.from_numpy(x_test_og).float()
  y_test       = torch.from_numpy(y_test_og.T).float()
  test_set     = TensorDataset(x_test, y_test)
  test_loader  = DataLoader(test_set, batch_size=32)
  return train_loader, test_loader

#### Define model ####

class HiddenLayer(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()

    self.hidden = nn.Sequential(
      nn.Linear(in_features, out_features, bias=True),
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
      nn.ReLU(),
    )

    self.hidden_layer = nn.Sequential(*[HiddenLayer(in_features=hidden_units, out_features=hidden_units) for _ in range(n_layer)])

    self.out_layer = nn.Sequential(
      nn.Linear(hidden_units, out_features, bias=True),
      nn.Sigmoid()
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
      test_pred  = model(x)
      test_loss += loss_fn(test_pred, y)
      test_acc  += acc_fn(test_pred, y)

  test_loss = test_loss / len(test_loader)
  test_acc  = test_acc / len(test_loader)
  return test_loss, test_acc

#### Define training and testing steps ####

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

def train(model, train_loader, test_loader, optimizer, loss_fn, acc_fn, epochs):
  for epoch in range(epochs):
    train_loss, train_acc = train_step(model, train_loader, optimizer, loss_fn, acc_fn)
    test_loss , test_acc  = test_step(model, test_loader, loss_fn, acc_fn)

    if epoch % 10 == 0:
      print(f"[{epoch}/{epochs}] Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f} || Test loss: {test_loss:.3f} | Test acc: {test_acc:.3f}")

if __name__ == "__main__":
  torch.manual_seed(0)
  train_loader, test_loader = load_dataset()
  in_features  = 64*64*3
  out_features = 1
  hidden_units = 5
  n_layer      = 3

  model     = Net(in_features, out_features, hidden_units, n_layer)
  loss_fn   = nn.BCELoss()
  acc_fn    = Accuracy(task='binary', num_classes=out_features)

  # Try out different learning rates
  lrs = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.01, 0.03, 0.1]
  epochs = 100
  for lr in lrs:
    print(f'lr: {lr}')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train(model, train_loader, test_loader, optimizer, loss_fn, acc_fn, epochs)