import torch
import torch.nn as nn
import torchmetrics
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import lightning as L
from centerloss.model import CLModel
from centerloss.loss import CenterLoss
import wandb
import matplotlib.pyplot as plt
import random


class CLModlit(L.LightningModule):
    def __init__(self, lamda):
        super().__init__()
        self.model = CLModel()
        self.softmax_loss = nn.CrossEntropyLoss()
        self.center_loss = CenterLoss()
        self.lamda = lamda
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10)
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer1 = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        optimizer2 = torch.optim.SGD(self.center_loss.parameters(), lr=0.5)

        return [optimizer1, optimizer2]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        opt = self.optimizers()
        label, coord = self.model(x)
        coords = coord.detach().cpu().numpy()
        labels = y.detach().cpu().numpy()
        plt.figure(figsize=(4, 4))
        plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=50, cmap='hsv')
        plt.xlabel('x')
        plt.ylabel('y')
        wandb.log({"train distribution": plt})
        plt.close()

        softmax_loss = self.softmax_loss(label, y)
        center_loss = self.center_loss(coord, y)
        loss = softmax_loss + self.lamda * center_loss

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.log('train loss', loss)
        self.log('train softmax loss', softmax_loss)
        self.log('train center loss', center_loss)

        return softmax_loss, center_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        label, coord = self.model(x)
        coords = coord.detach().cpu().numpy()
        labels = y.detach().cpu().numpy()
        plt.figure(figsize=(4, 4))
        plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=50, cmap='hsv')
        plt.xlabel('x')
        plt.ylabel('y')
        wandb.log({"valid distribution": plt})
        plt.close()

        softmax_loss = self.softmax_loss(label, y)
        center_loss = self.center_loss(coord, y)
        loss = softmax_loss + self.lamda * center_loss
        self.log('validation loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        label, coord = self.model(x)
        loss = self.softmax_loss(label, y) + self.lamda*self.center_loss(coord, y)
        correct_pred = torch.argmax(label, dim=1)
        acc = self.accuracy(correct_pred, y)
        self.log('test accuracy', acc)
