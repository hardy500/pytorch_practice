from torch import nn
from torchsummary import summary

class CNN_0(nn.Module):
  # [ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL] -> FLATTEN -> DENSE
  def __init__(self, num_classes):
    super().__init__()

    self.features = nn.Sequential(
      nn.ZeroPad2d((3, 3)),                      ## Conv2D with 32 7x7 filters and stride of 1
      nn.Conv2d(3, 32, kernel_size=7, stride=1), ## Conv2D with 32 7x7 filters and stride of 1
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2),     ## Max Pooling 2D with default parameters
      )

    self.classifier = nn.Sequential(
      nn.Flatten(),
      ## Dense layer with 1 unit for output & 'sigmoid' activation
      nn.Linear(29696, num_classes),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = nn.functional.normalize(x)
    x = self.features(x)
    x = self.classifier(x)
    return x


class CNN_1(nn.Module):
  # [CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL] -> FLATTEN -> DENSE
  def __init__(self, num_classes):
    super().__init__()

    self.features = nn.Sequential(
      nn.Conv2d(3, 16, kernel_size=4, padding=1, stride=1),
      nn.BatchNorm2d(16), # this help to git the model better
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=8, stride=8, padding=0),

      nn.Conv2d(16, 2, kernel_size=2, padding=1, stride=1),
      nn.BatchNorm2d(2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
    )

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(8, num_classes),
    )

  def forward(self, x):
    x      = nn.functional.normalize(x)
    x      = self.features(x)
    logits = self.classifier(x)
    return logits