import argparse
import numpy as np
import pickle

import torch
import torchvision

import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import DataLoader
from lr_finder import LRFinder

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
parser.add_argument("-d", "--dirname", type=str, default="chanelset-bags", help="dataset dirname in split_dataset directory")
args = parser.parse_args()


def load_dataset(image_size, batch_size, train_dir, test_dir, dirname):

    train_transforms = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    train_data = torchvision.datasets.ImageFolder(
        root=train_dir, transform=train_transforms
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    mean, std = normalization_parameter(train_loader)

    # image transformations for train and test data
    train_transforms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
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
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # inverse normalization for image plot
    inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=1 / std)

    # data loader
    train_data = torchvision.datasets.ImageFolder(
        root=train_dir, transform=train_transforms
    )
    test_data = torchvision.datasets.ImageFolder(
        root=test_dir, transform=test_transforms
    )
    dataloaders = data_loader(
        train_data, test_data, valid_size=0.2, batch_size=batch_size
    )
    # label of classes
    classes = train_data.classes
    decoder = {}
    for i in range(len(classes)):
        decoder[classes[i]] = i
    encoder = {}
    for i in range(len(classes)):
        encoder[i] = classes[i]
    os.makedirs(f'../trained_models/{dirname}/bin/')
    class_plot(train_data, encoder, inv_normalize, output_dir=dirname)

    with open(f"../trained_models/{dirname}/bin/encoder.pickle","wb") as enc:
        pickle.dump(encoder, enc)
    with open(f"../trained_models/{dirname}/bin/test_transforms.pickle","wb") as trn:
        pickle.dump(test_transforms, trn)
    with open(f"../trained_models/{dirname}/bin/inv_normalize.pickle","wb") as inv:
        pickle.dump(inv_normalize, inv)

    return train_loader, dataloaders, classes, encoder, inv_normalize


def declare_model(train_loader, num_class, find_lr=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = Classifier(num_class=num_class).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(classifier.parameters(), lr=0.0000001)
    if find_lr:
        print('Running LRFinder. . .')
        lr_finder = LRFinder(classifier, optimizer_ft, criterion, device=device)
        lr_finder.range_test(train_loader, end_lr=1, num_iter=500)
        lr_finder.reset()
        lr_finder.plot()

    return classifier, criterion


if __name__ == "__main__":
    batch_size = args.batchsize
    image_size = args.imagesize
    dirname = args.dirname
    train_dir = f"../../split_dataset/{dirname}/train/"
    test_dir = f"../../split_dataset/{dirname}/val/"
    train_loader, dataloaders, classes, encoder, inv_normalize = load_dataset(
        image_size=image_size,
        batch_size=batch_size,
        train_dir=f"../../split_dataset/{dirname}/train/",
        test_dir=f"../../split_dataset/{dirname}/val/",
        dirname=dirname
    )
    classifier, criterion = declare_model(train_loader, len(classes), find_lr=False)

    train_model(
        classifier,
        dataloaders,
        criterion,
        patience=3,
        batch_size=batch_size,
        classes=classes,
        encoder=encoder,
        inv_normalize=inv_normalize,
        output_dir=dirname
    )
