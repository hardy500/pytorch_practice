import torch
from torch import nn

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

def train_step(model, train_loader, optimizer, loss_fn, acc_fn):
  train_loss, train_acc = 0, 0
  model.train()
  for x, y in train_loader:
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
  # Training
  for epoch in range(epochs+1):
    train_loss, train_acc = train_step(model, train_loader, optimizer, loss_fn, acc_fn)
    print(f"[{epoch}/{epochs}] Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f}")

  # Testing
  test_loss, test_acc = test_step(model, test_loader, loss_fn, acc_fn)
  print(f"Test loss: {test_loss:.3f} | Test acc: {test_acc:.3f}")