import pickle
import numpy as np
from PIL import Image
import cv2
import torch
from torch import nn
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--modelpath", type=str, default="../trained_models/watches_acc_96", help="path to folder containing model.pth and bin folder")
args = vars(parser.parse_args())

MODEL_PATH = args["modelpath"]

def predict(image, transforms, encoder):
    model = torch.load(MODEL_PATH + '/model.pth')
    model.eval()
    if(isinstance(image,np.ndarray)):
      image = Image.fromarray(image)
    if(transforms!=None):
        image = transforms(image)
    data = image.expand(1,-1,-1,-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.type(torch.FloatTensor).to(device)
    sm = nn.Softmax(dim = 1)
    output = model(data)
    output = sm(output)
    _, preds = torch.max(output, 1)

    pred_label = encoder[preds.cpu().detach().numpy()[0]]
    return pred_label

if __name__ == "__main__":
    with open(MODEL_PATH + "/bin/encoder.pickle", 'rb') as fr:
        encoder = pickle.load(fr)
    with open(MODEL_PATH + "/bin/test_transforms.pickle", 'rb') as fr:
        transforms = pickle.load(fr)
    with open(MODEL_PATH + "/bin/inv_normalize.pickle", 'rb') as fr:
        inv_normalize = pickle.load(fr)

    image = cv2.imread('../sample_images/watch_codecoco.jpg')
    pred_label = predict(image,transforms,encoder)
    print(pred_label)