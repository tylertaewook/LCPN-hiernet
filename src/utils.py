import numpy as np
import random
import torch
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics

matplotlib.style.use("ggplot")

# plotting rondom images from dataset
def class_plot(data, encoder, inv_normalize=None, n_figures=12):
    n_row = int(n_figures / 4)
    fig, axes = plt.subplots(figsize=(14, 10), nrows=n_row, ncols=4)
    for ax in axes.flatten():
        a = random.randint(0, len(data))
        (image, label) = data[a]
        print(type(image))
        label = int(label)
        l = encoder[label]
        if inv_normalize != None:
            image = inv_normalize(image)

        image = image.numpy().transpose(1, 2, 0)
        im = ax.imshow(image)
        ax.set_title(l)
        ax.axis("off")
    print("plot saved: /outputs/class_plot.png")
    plt.savefig("../outputs/class_plot.png")


def error_plot(loss):
    plt.figure(figsize=(10, 5))
    plt.plot(loss)
    plt.title("Training loss plot")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    print("plot saved: /outputs/error_plot.png")
    plt.savefig("../outputs/error_plot.png")


def acc_plot(acc):
    plt.figure(figsize=(10, 5))
    plt.plot(acc)
    plt.title("Training accuracy plot")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    print("plot saved: /outputs/acc_plot.png")
    plt.savefig("../outputs/acc_plot.png")


# To plot the wrong predictions given by model
def wrong_plot(n_figures, true, ima, pred, encoder, inv_normalize):
    print("Classes in order Actual and Predicted")
    n_row = int(n_figures / 3)
    fig, axes = plt.subplots(figsize=(14, 10), nrows=n_row, ncols=3)
    for ax in axes.flatten():
        a = random.randint(0, len(true) - 1)

        image, correct, wrong = ima[a], true[a], pred[a]
        image = torch.from_numpy(image)
        correct = int(correct)
        c = encoder[correct]
        wrong = int(wrong)
        w = encoder[wrong]
        f = "A:" + c + "," + "P:" + w
        if inv_normalize != None:
            image = inv_normalize(image)
        image = image.numpy().transpose(1, 2, 0)
        im = ax.imshow(image)
        ax.set_title(f)
        ax.axis("off")
    print("plot saved: /outputs/wrong_plot.png")
    plt.savefig("../outputs/wrong_plot.png")


def plot_confusion_matrix(
    y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # ax.set_xticks(np.arange(cm.shape[1]+1)-.5)
    # ax.set_yticks(np.arange(cm.shape[0]+1)-.5)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    print("plot saved: /outputs/confusion_matrix.png")
    plt.savefig("../outputs/confusion_matrix.png")
    return ax


def performance_matrix(true, pred):
    precision = metrics.precision_score(true, pred, average="macro")
    recall = metrics.recall_score(true, pred, average="macro")
    accuracy = metrics.accuracy_score(true, pred)
    f1_score = metrics.f1_score(true, pred, average="macro")
    print(
        "Precision: {} Recall: {}, Accuracy: {}: ,f1_score: {}".format(
            precision * 100, recall * 100, accuracy * 100, f1_score * 100
        )
    )
    f= open("../outputs/performance.txt","w+")
    f.write("Precision: {} Recall: {}, Accuracy: {}: ,f1_score: {}".format(
        precision * 100, recall * 100, accuracy * 100, f1_score * 100
    ))
    f.close()

