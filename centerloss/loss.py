import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2):
        super(CenterLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.center = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, pred, labels):
        centers = self.center[labels]
        loss = self.mse_loss(pred, centers)
        loss /= labels.shape[0]
        loss /= 2

        return loss
