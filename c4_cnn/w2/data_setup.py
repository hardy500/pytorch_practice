from cnn_utils import load_happy_dataset, load_signs_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import Accuracy

def load_data(batch_size, dataset=None):
  if dataset == "happy_dataset":
    x_train_og, y_train_og, x_test_og, y_test_og, classes = load_happy_dataset()
  elif dataset == "signs_dataset":
    x_train_og, y_train_og, x_test_og, y_test_og, classes= load_signs_dataset()
  else:
    print("Missing 'dataset' argument")

  # (N, C, H, W) <- this size needed to do cnn in pytorch
  x_train      = torch.from_numpy(x_train_og).permute(0, 3, 1, 2).float()
  y_train      = torch.from_numpy(y_train_og.T).float()
  train_set    = TensorDataset(x_train, y_train)
  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

  x_test      = torch.from_numpy(x_test_og).permute(0, 3, 1, 2).float()
  y_test      = torch.from_numpy(y_test_og.T).float()
  test_set    = TensorDataset(x_train, y_train)
  test_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
  return train_loader, test_loader, classes