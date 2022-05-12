import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
from efficientnet_pytorch import EfficientNet
from lr_finder import LRFinder
import torch.optim as optim

# using efficientnet model based transfer learning
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.resnet = EfficientNet.from_pretrained("efficientnet-b0")
        self.l1 = nn.Linear(1000, 256)
        self.dropout = nn.Dropout(0.75)
        self.l2 = nn.Linear(256, 19)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.resnet(input)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return x


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), "checkpoint.pt")
        self.val_loss_min = val_loss


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     classifier = Classifier().to(device)

#     # Loss -> Negative log likelihood loss if output layer logsoftmax else for linear layer we use crossentropy loss.
#     criterion = nn.CrossEntropyLoss()
#     # lr scheduler ->
#     # learning rate half after 3 epochs
#     # cyclical learning rate ->
#     # Original learning rate restored after 10 epochs

#     optimizer_ft = optim.Adam(classifier.parameters(), lr=0.0000001)
#     lr_finder = LRFinder(classifier, optimizer_ft, criterion, device=device)
#     lr_finder.range_test(train_loader, end_lr=1, num_iter=500)
#     lr_finder.reset()
#     lr_finder.plot()
