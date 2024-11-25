import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from centerloss.modlit import CLModlit
import argparse
import numpy as np
from PIL import Image


def main():

    # define parser
    parser = argparse.ArgumentParser()

    # load trained model
    checkpoint = "./model/lenet.ckpt"
    model = CLModlit.load_from_checkpoint(checkpoint, lamda=0.01)

    #########################################################################################################
    # ========================================== data preprocess ========================================== #
    #########################################################################################################
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor()
    ])

    # using your own data
    parser.add_argument('-img', type=str, help='Input Image Path')
    img_path = parser.parse_args().img
    img = Image.open(img_path)
    image = img.resize((28, 28)).convert('L')
    img_tensor = transforms.ToTensor()(image)
    img_tensor = img_tensor.unsqueeze(0).to('cuda')

    #########################################################################################################
    # ============================================== process ============================================== #
    #########################################################################################################
    # put inference img tensor into the prediction model
    result = model(img_tensor)

    #########################################################################################################
    # ============================================ postprocess ============================================ #
    #########################################################################################################
    # using your own data
    result = result[0].squeeze(0).detach().to('cpu').numpy()
    res = np.argmax(result)
    print(res)


if __name__ == '__main__':
    main()
