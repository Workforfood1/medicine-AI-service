import torch
import torch.nn as nn

class MainModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 128)
        self.conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.last_layer = nn.Linear(256 * 16 * 16, 10)  # Assuming input images are 32x32

    def call(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.last_layer(x)
        return x
    


