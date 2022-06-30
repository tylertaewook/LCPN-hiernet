import argparse
import pickle
import numpy as np
from PIL import Image
import cv2
import torch
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--imagepath", type=str, default="../sample_images/watch_j12.jpg", help="Path to image you wish to predict on")
args = parser.parse_args()

def predict(modelpath, image, transforms, encoder):
    print("Loading model.pth...")
    model = torch.load(modelpath + '/model.pth')
    model.eval()
    if(isinstance(image,np.ndarray)):
      image = Image.fromarray(image)
    if(transforms!=None):
        image = transforms(image)
    data = image.expand(1,-1,-1,-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Forward Passing...")
    data = data.type(torch.FloatTensor).to(device)
    sm = nn.Softmax(dim = 1)
    output = model(data)
    output = sm(output)
    print("Predicting Label...")
    _, preds = torch.max(output, 1)

    pred_label = encoder[preds.cpu().detach().numpy()[0]]
    return pred_label

if __name__ == "__main__":
    print("=======Phase 1: Parent Class Prediction========")
    modelpath = "../trained_models/parent"

    print("Loading pkl files...")
    with open(modelpath + "/bin/encoder.pickle", 'rb') as fr:
        encoder = pickle.load(fr)
    with open(modelpath + "/bin/test_transforms.pickle", 'rb') as fr:
        transforms = pickle.load(fr)
    with open(modelpath + "/bin/inv_normalize.pickle", 'rb') as fr:
        inv_normalize = pickle.load(fr)
    print("ENCODER: ", encoder)
    image = cv2.imread(args.imagepath)
    parent_pred_label = predict(modelpath, image,transforms,encoder)

    parent_pred_class = parent_pred_label.split('_', 1)[0].replace('.', '') # watch, cosm, perf, etc.

    print("Parent class prediction: ", parent_pred_class)
    child_modelpath = "../trained_models/" + parent_pred_class
    print("=======Phase 2: Child Class Prediction========")

    print("Loading pkl files...")
    with open(child_modelpath + "/bin/encoder.pickle", 'rb') as fr:
        encoder = pickle.load(fr)
    with open(child_modelpath + "/bin/test_transforms.pickle", 'rb') as fr:
        transforms = pickle.load(fr)
    with open(child_modelpath + "/bin/inv_normalize.pickle", 'rb') as fr:
        inv_normalize = pickle.load(fr)
    print("ENCODER: ", encoder)
    child_pred_label = predict(child_modelpath, image, transforms,encoder)


    print("Final Prediction: ", child_pred_label)