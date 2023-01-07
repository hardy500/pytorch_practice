from data_setup import load_data
from engine_0 import train
from model import CNN_0

import torch
from torch import nn
from torchmetrics import Accuracy

### GOAL: Recognize smile vs no smily image using CNN

train_loader, test_loader, classes = load_data(32)

N_CLASSES = 1
LR        = 0.001
EPOCHS    = 10

model     = CNN_0(N_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn   = nn.BCELoss()
acc_fn    = Accuracy(task='binary', num_classes=N_CLASSES)

torch.manual_seed(0)
train(model, train_loader, test_loader, optimizer, loss_fn, acc_fn, EPOCHS)
# result: [10/10] Train loss: 0.035 | Train acc: 0.995 Test loss: 0.044 | Test acc: 0.991