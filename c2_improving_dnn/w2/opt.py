import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import Accuracy
from opt_utils import load_dataset
from torch.optim.lr_scheduler import StepLR

# GOAL: Overfit training data using optimization technique

#### Preprocessing data ####
# size (2, 300), (1, 300)
def load_data():
  x_train_og, y_train_og = load_dataset()
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
    #x = nn.functional.normalize(x) # NOTE: This model seem to train better without this
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
    optimizer.zero_grad()

    pred        = model(x)
    loss        = loss_fn(pred, y)
    train_loss += loss
    train_acc  += acc_fn(pred, y)

    loss.backward()
    optimizer.step()

  train_loss /= len(train_loader)
  train_acc  /= len(train_loader)
  return train_loss, train_acc

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

def train(model, train_loader, optimizer, scheduler, loss_fn, acc_fn, epochs):
  for epoch in range(epochs+1):
    train_loss, train_acc = train_step(model, train_loader, optimizer, loss_fn, acc_fn)
    scheduler.step()

    if epoch % 10 == 0:
      print(f"[{epoch}/{epochs}] Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f}")

if __name__ == "__main__":
  torch.manual_seed(0)
  train_loader = load_data()

  lambd        = 0.0000000# hyperparameter for l2 reg
  momentum     = 0.9
  in_features  = 2
  out_features = 1
  hidden_units = 5
  n_layer      = 3

  model     = Net(in_features, out_features, hidden_units, n_layer)
  loss_fn   = nn.BCELoss()
  acc_fn    = Accuracy(task='binary', num_classes=out_features)
  # NOTE:
  # Adam will take >2x epochs to train then SGD with momentum
  # Without momentum SGD will perform worse then Adam with the same number of epochs
  optimizer = torch.optim.SGD(model.parameters(), lr=0.15, momentum=momentum, weight_decay=lambd)
  #optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=lambd)

  # NOTE: By using lr decay, we decrease the number of epochs by 40%
  # while maintaining the same accuracy (95%)
  scheduler = StepLR(optimizer, step_size=100)

  epochs       = 140
  train(model, train_loader, optimizer, scheduler, loss_fn, acc_fn, epochs)