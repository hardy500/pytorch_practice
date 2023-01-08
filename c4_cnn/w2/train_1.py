from data_setup import load_data
from engine_1 import train
from torchmetrics import Accuracy
from model import CNN_1
import torch
from torch import nn

train_loader, test_loader, classes = load_data(64, dataset="signs_dataset")

N_CLASSES = len(classes)
LR = 0.01

model = CNN_1(N_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn   = nn.CrossEntropyLoss()
acc_fn    = Accuracy(task='multiclass', num_classes=N_CLASSES)

EPOCHS = 100
train(model, train_loader, test_loader, optimizer, loss_fn, acc_fn, EPOCHS)