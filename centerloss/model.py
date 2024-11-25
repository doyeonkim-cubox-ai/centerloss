import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class CLModel(nn.Module):
    def __init__(self):
        super(CLModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2, stride=1, bias=False)
        self.conv1_1 = nn.Conv2d(32, 32, 5, padding=2, stride=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2, stride=1, bias=False)
        self.conv2_1 = nn.Conv2d(64, 64, 5, padding=2, stride=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2, stride=1, bias=False)
        self.conv3_1 = nn.Conv2d(128, 128, 5, padding=2, stride=1, bias=False)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.prelu = nn.PReLU()
        self.fc1 = nn.Linear(1152, 2, bias=False)
        self.fc2 = nn.Linear(2, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu(x)
        x = self.conv1_1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.prelu(x)
        x = self.conv2_1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.prelu(x)
        x = self.conv3_1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.prelu(x)
        y = self.fc2(x)

        return x, y


# model = CLModel()
# x = torch.randn(1, 1, 28, 28)
# res = model(x)
