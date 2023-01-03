from lr_utils import load_dataset
import torch
from torch import nn
from torchmetrics.classification import Accuracy
from torch.utils.data import DataLoader, TensorDataset

#### Preprocesing data ####
# input size x (N, H, W, C), input size y (1, N)
x_train_og, y_train_og, x_test_og, y_test_og, classes = load_dataset()

x_train = torch.from_numpy(x_train_og).float()
y_train = torch.from_numpy(y_train_og.T).float()

x_test = torch.from_numpy(x_test_og).float()
y_test = torch.from_numpy(y_test_og.T).float()

train_set    = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

test_set    = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_set, batch_size=1)

#### Define model ####

class LR(nn.Module):
  def __init__(self, in_features, out_features):
    super().__init__()

    self.layer_stack = nn.Sequential(
      nn.Flatten(),
      nn.Linear(in_features, out_features),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = nn.functional.normalize(x)
    x = self.layer_stack(x)
    return x

def test_step(model, test_loader, loss_fn, acc_fn):
  test_loss, test_acc  = 0, 0
  model.eval()
  with torch.inference_mode():
    for x, y in test_loader:
      test_pred  = model(x)
      test_loss += loss_fn(test_pred, y)
      test_acc  += acc_fn(test_pred, y)

    test_loss /= len(test_loader)
    test_acc  /= len(test_loader)
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
  in_features  = 64*64*3
  out_features = 1

  model     = LR(in_features, out_features)
  loss_fn   = nn.BCELoss()
  #optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
  optimizer = torch.optim.Adam(model.parameters())
  acc_fn    = Accuracy(task='binary', num_classes=1)
  epochs    = 100

  torch.manual_seed(0)

  ##### Traning ####
  train(model, train_loader, test_loader, optimizer, loss_fn, acc_fn, epochs)