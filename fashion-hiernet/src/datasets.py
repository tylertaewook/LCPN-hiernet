import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm


def normalization_parameter(dataloader):
    mean = 0.0
    std = 0.0
    nb_samples = len(dataloader.dataset)
    for data, _ in tqdm(dataloader):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
    mean /= nb_samples
    std /= nb_samples
    return mean.numpy(), std.numpy()


def data_loader(train_data, test_data=None, valid_size=None, batch_size=32):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    if test_data == None and valid_size == None:
        dataloaders = {"train": train_loader}
        return dataloaders
    if test_data == None and valid_size != None:
        data_len = len(train_data)
        indices = list(range(data_len))
        np.random.shuffle(indices)
        split1 = int(np.floor(valid_size * data_len))
        valid_idx, test_idx = indices[:split1], indices[split1:]
        valid_sampler = SubsetRandomSampler(valid_idx)
        valid_loader = DataLoader(
            train_data, batch_size=batch_size, sampler=valid_sampler
        )
        dataloaders = {"train": train_loader, "val": valid_loader}
        return dataloaders
    if test_data != None and valid_size != None:
        data_len = len(test_data)
        indices = list(range(data_len))
        np.random.shuffle(indices)
        split1 = int(np.floor(valid_size * data_len))
        valid_idx, test_idx = indices[:split1], indices[split1:]
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        valid_loader = DataLoader(
            test_data, batch_size=batch_size, sampler=valid_sampler
        )
        test_loader = DataLoader(test_data, batch_size=batch_size, sampler=test_sampler)
        dataloaders = {"train": train_loader, "val": valid_loader, "test": test_loader}
        return dataloaders