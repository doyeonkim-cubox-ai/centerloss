import torch
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import lightning as L
from centerloss.modlit import CLModlit
from centerloss.data import CLDataModule
import argparse


def main():
    dm = CLDataModule(data_dir="./mnist", batch_size=128)

    checkpoint = "./model/lenet.ckpt"
    net = CLModlit.load_from_checkpoint(checkpoint, lamda=0.01)
    trainer = L.Trainer(accelerator='cuda', devices=1)
    trainer.test(net, dm)


if __name__ == "__main__":
    main()
