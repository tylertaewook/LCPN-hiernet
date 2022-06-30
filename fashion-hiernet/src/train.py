import torch
import argparse
import torch.optim as optim
import time
import datetime
import numpy as np

from tqdm.auto import tqdm
from model import EarlyStopping
from torch.autograd import Variable
from torch import nn

from utils import (
    error_plot,
    wrong_plot,
    plot_confusion_matrix,
    performance_matrix,
)


def train(
    model,
    dataloaders,
    criterion,
    num_epochs=10,
    lr=0.00001,
    batch_size=8,
    patience=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    since = time.time()
    model.to(device)
    best_acc = 0.0
    i = 0
    phase1 = dataloaders.keys()
    losses = list()
    acc = list()
    if patience != None:
        earlystop = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        lr = lr * 0.8
        if epoch % 10 == 0:
            lr = 0.0001

        for phase in phase1:
            if phase == " train":
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            total = 0
            j = 0
            for batch_idx, (data, target) in enumerate(dataloaders[phase]):
                data, target = Variable(data), Variable(target)
                data = data.type(torch.cuda.FloatTensor)
                target = target.type(torch.cuda.LongTensor)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                _, preds = torch.max(output, 1)
                running_corrects = running_corrects + torch.sum(preds == target.data)
                running_loss += loss.item() * data.size(0)
                j = j + 1
                if phase == "train":
                    loss.backward()
                    optimizer.step()

                if batch_idx % 300 == 0:
                    print(
                        "{} Epoch: {}  [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAcc: {:.6f}".format(
                            phase,
                            epoch,
                            batch_idx * len(data),
                            len(dataloaders[phase].dataset),
                            100.0 * batch_idx / len(dataloaders[phase]),
                            running_loss / (j * batch_size),
                            running_corrects.double() / (j * batch_size),
                        )
                    )
            epoch_acc = running_corrects/(len(dataloaders[phase]) * batch_size)
            epoch_loss = running_loss / (len(dataloaders[phase]) * batch_size)
            if phase == "val":
                earlystop(epoch_loss, model)

            if phase == "train":
                losses.append(epoch_loss)
                acc.append(epoch_acc)
            print(earlystop.early_stop)
        if earlystop.early_stop:
            print("Early stopping")
            model.load_state_dict(torch.load("./checkpoint.pt"))
            break
        print("{} Accuracy: ".format(phase), epoch_acc.item())
    runtime = str(datetime.timedelta(seconds=time.time() - since)).split('.')[0]
    print("Training Time: ", runtime)
    return losses, acc


def train_model(
    model,
    dataloaders,
    criterion,
    num_epochs=10,
    lr=0.0001,
    batch_size=8,
    patience=None,
    classes=None,
    encoder=None,
    inv_normalize=None,
    output_dir=None,
):
    dataloader_train = {}
    losses = list()
    key = dataloaders.keys()
    for phase in key:
        if phase == "test":
            perform_test = True
        else:
            dataloader_train.update([(phase, dataloaders[phase])])
    losses, _ = train(
        model, dataloader_train, criterion, num_epochs, lr, batch_size, patience
    )
    error_plot(losses, output_dir=output_dir)
    if perform_test == True:
        true, pred, image, true_wrong, pred_wrong = test(dataloaders["test"], model=model, criterion=criterion, batch_size=batch_size)
        wrong_plot(12, true_wrong, image, pred_wrong, encoder, inv_normalize, output_dir=output_dir)
        performance_matrix(true, pred, output_dir=output_dir)
        if classes != None:
            plot_confusion_matrix(
                true,
                pred,
                classes=classes,
                title=f"Confusion matrix: {output_dir}",
                output_dir=output_dir
            )
    torch.save(model, f'../trained_models/{output_dir}/model.pth')


def test(dataloader, model=None, criterion=None, batch_size=8):
    running_corrects = 0
    running_loss = 0
    pred = []
    true = []
    pred_wrong = []
    true_wrong = []
    image = []
    sm = nn.Softmax(dim=1)
    classifier = model
    for _, (data, target) in enumerate(dataloader):
        data, target = Variable(data), Variable(target)
        data = data.type(torch.cuda.FloatTensor)
        target = target.type(torch.cuda.LongTensor)
        classifier.eval()
        output = classifier(data)
        loss = criterion(output, target)
        output = sm(output)
        _, preds = torch.max(output, 1)
        running_corrects = running_corrects + torch.sum(preds == target.data)
        running_loss += loss.item() * data.size(0)
        preds = preds.cpu().numpy()
        target = target.cpu().numpy()
        preds = np.reshape(preds, (len(preds), 1))
        target = np.reshape(target, (len(preds), 1))
        data = data.cpu().numpy()

        for i in range(len(preds)):
            pred.append(preds[i])
            true.append(target[i])
            if preds[i] != target[i]:
                pred_wrong.append(preds[i])
                true_wrong.append(target[i])
                image.append(data[i])

    epoch_acc = running_corrects.double()/(len(dataloader)*batch_size)
    epoch_loss = running_loss / (len(dataloader) * batch_size)
    print(epoch_acc, epoch_loss)
    return true, pred, image, true_wrong, pred_wrong
