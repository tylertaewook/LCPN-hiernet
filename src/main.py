import numpy as np
import pickle

import torch
import torchvision

import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import DataLoader
from lr_finder import LRFinder

import argparse

from datasets import normalization_parameter, data_loader
from model import Classifier
from train import train_model
from utils import class_plot
import os

# To see RuntimeError msg; device-side assert riggered
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batchsize", type=int, default=8, help="Batch size")
parser.add_argument("-i", "--imagesize", type=int, default=150, help="Image size")
parser.add_argument("-tr", "--traindir", type=str, default="../../chxnelset_split/train/", help="Train Directory")
parser.add_argument("-te", "--testdir", type=str, default="../../chxnelset_split/val/", help="Test Directory")
args = vars(parser.parse_args())

BATCH_SIZE = args["batchsize"]
IM_SIZE = args["imagesize"]


def load_dataset():

    train_transforms = transforms.Compose(
        [transforms.Resize((IM_SIZE, IM_SIZE)), transforms.ToTensor()]
    )
    train_data = torchvision.datasets.ImageFolder(
        root=args["traindir"], transform=train_transforms
    )
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    mean, std = normalization_parameter(train_loader)

    # image transformations for train and test data

    train_transforms = transforms.Compose(
        [
            transforms.Resize((IM_SIZE, IM_SIZE)),
            transforms.RandomResizedCrop(size=315, scale=(0.95, 1.0)),
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=299),  # Image net standards
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize((IM_SIZE, IM_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # inverse normalization for image plot

    inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=1 / std)

    # data loader
    train_data = torchvision.datasets.ImageFolder(
        root=args["traindir"], transform=train_transforms
    )
    test_data = torchvision.datasets.ImageFolder(
        root=args["testdir"], transform=test_transforms
    )
    dataloaders = data_loader(
        train_data, test_data, valid_size=0.2, batch_size=BATCH_SIZE
    )
    # label of classes
    classes = train_data.classes
    # encoder and decoder to convert classes into integer
    decoder = {}
    for i in range(len(classes)):
        decoder[classes[i]] = i
    encoder = {}
    for i in range(len(classes)):
        encoder[i] = classes[i]

    class_plot(train_data, encoder, inv_normalize)

    with open("../bin/encoder.pickle","wb") as enc:
        pickle.dump(encoder, enc)
    with open("../bin/test_transforms.pickle","wb") as trn:
        pickle.dump(test_transforms, trn)
    with open("../bin/inv_normalize.pickle","wb") as inv:
        pickle.dump(inv_normalize, inv)

    return train_loader, dataloaders, classes, encoder, inv_normalize


def declare_model(train_loader, num_class):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = Classifier(num_class=num_class).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(classifier.parameters(), lr=0.0000001)
    # print('Running LRFinder. . .')
    # lr_finder = LRFinder(classifier, optimizer_ft, criterion, device=device)
    # lr_finder.range_test(train_loader, end_lr=1, num_iter=500)
    # lr_finder.reset()
    # lr_finder.plot()

    return classifier, criterion


if __name__ == "__main__":
    train_loader, dataloaders, classes, encoder, inv_normalize = load_dataset()
    classifier, criterion = declare_model(train_loader, len(classes))

    train_model(
        classifier,
        dataloaders,
        criterion,
        patience=3,
        batch_size=BATCH_SIZE,
        classes=classes,
        encoder=encoder,
        inv_normalize=inv_normalize
    )
