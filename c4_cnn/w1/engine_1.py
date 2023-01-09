import torch
from torch import nn

def test_step(model, test_loader, loss_fn, acc_fn):
  test_loss, test_acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for x, y in test_loader:
      logits      = model(x)
      pred        = nn.functional.softmax(logits, dim=1)
      loss        = loss_fn(logits, y.squeeze().long())
      test_loss += loss.item()
      test_acc  += acc_fn(pred.argmax(1), y.squeeze())

  test_loss = test_loss / len(test_loader)
  test_acc  = test_acc  / len(test_loader)
  return test_loss, test_acc

def train_step(model, train_loader, optimizer, loss_fn, acc_fn):
  train_loss, train_acc = 0, 0
  model.train()
  for x, y in train_loader:
    optimizer.zero_grad()

    logits      = model(x)
    pred        = nn.functional.softmax(logits, dim=1)
    loss        = loss_fn(logits, y.squeeze().long())
    train_loss += loss.item()
    train_acc  += acc_fn(pred.argmax(1), y.squeeze())

    loss.backward()
    optimizer.step()

  train_loss /= len(train_loader)
  train_acc  /= len(train_loader)
  return train_loss, train_acc

def train(model, train_loader, test_loader, optimizer, loss_fn, acc_fn, epochs):
  for epoch in range(epochs+1):
    train_loss, train_acc = train_step(model, train_loader, optimizer, loss_fn, acc_fn)
    test_loss, test_acc   = test_step(model, test_loader, loss_fn, acc_fn)
    if epoch % 10 == 0:
      print(f"[{epoch}/{epochs}] Train loss: {train_loss:.3f} | Train acc: {train_acc:.3f}\
            || Test loss: {test_loss:.3f} | Test acc: {test_acc:.3f}")