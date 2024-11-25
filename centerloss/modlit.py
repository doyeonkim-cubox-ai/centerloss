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
import numpy as np


class CLModlit(L.LightningModule):
    def __init__(self, lamda):
        super().__init__()
        self.model = CLModel()
        self.softmax_loss = nn.CrossEntropyLoss()
        self.center_loss = CenterLoss()
        self.lamda = lamda
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=10)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        all_features, all_labels = [], []
        coord, label = self.model(x)
        all_features.append(coord.data.cpu().numpy())
        all_labels.append(label.data.cpu().numpy())
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        colors = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
        label_idx = all_labels.argmax(axis=1)
        for i in range(10):
            plt.scatter(all_features[label_idx == i, 0], all_features[label_idx == i, 1], color=colors[i], s=50)
        plt.xlabel('x')
        plt.ylabel('y')
        wandb.log({"train distribution": plt})
        plt.close()

        softmax_loss = self.softmax_loss(label, y)
        center_loss = self.center_loss(coord, y)
        loss = softmax_loss + self.lamda * center_loss

        self.log('train loss', loss)
        self.log('train softmax loss', softmax_loss)
        self.log('train center loss', center_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        all_features, all_labels = [], []
        coord, label = self.model(x)
        all_features.append(coord.data.cpu().numpy())
        all_labels.append(label.data.cpu().numpy())
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        colors = np.array(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
        label_idx = all_labels.argmax(axis=1)
        for i in range(10):
            plt.scatter(all_features[label_idx == i, 0], all_features[label_idx == i, 1], color=colors[i], s=50)
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
        coord, label = self.model(x)
        loss = self.softmax_loss(label, y) + self.lamda*self.center_loss(coord, y)
        correct_pred = torch.argmax(label, dim=1)
        acc = self.accuracy(correct_pred, y)
        self.log('test accuracy', acc)
