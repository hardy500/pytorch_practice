from torch import nn
from torchsummary import summary

class CNN_0(nn.Module):
  # ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
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