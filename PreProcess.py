import os
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import math
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

img_row, img_col, channels = 224, 224, 3
batch_size = 64
n_rows = 1000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = True

## classes_to_idx && idx_to_classes
class_files = os.listdir("./data/")
classes = {x[:-4]:i for i, x in enumerate(class_files)}
to_class = {i:x[:-4].replace(" ", "_") for i, x in enumerate(class_files)}

dfs = [pd.read_csv("./data/" + x, nrows=n_rows)[["word", "drawing"]] for x in class_files]
df = pd.concat(dfs)
del dfs

n_samples = df.shape[0]

pick_order = np.arange(n_samples)

def strokes_to_img(in_strokes):
    in_strokes = eval(in_strokes)
    # make an agg figure
    fig, ax = plt.subplots()
    for x, y in in_strokes:
        ax.plot(x, y, linewidth=12.)  # marker='.',
    ax.axis('off')
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    X = np.array(fig.canvas.renderer._renderer)
    # X = np.array(fig.canvas)
    plt.close(fig)
    return (cv2.resize(X, (img_row, img_col)) / 255.)[::-1]

def train_gen(phase):
    # while True:  # Infinity loop
    if phase == 'train':
        pick_per_epoch = n_samples // batch_size
    else:
        pick_per_epoch = int(n_samples * 0.1) // batch_size
    np.random.shuffle(pick_order)
    for i in range(pick_per_epoch):
        c_pick = pick_order[i*batch_size: (i+1)*batch_size]
        dfs = df.iloc[c_pick]
        out_imgs = list(map(strokes_to_img, dfs["drawing"]))
        X = np.array(out_imgs)[:, :, :, :channels].astype(np.float32)
        X = torch.tensor(X).permute(0, 3, 1, 2)
        y = np.array([classes[x] for x in dfs["word"]])
        y = torch.tensor(y)
        yield X, y

# data_loader = train_gen()

def display_img(items, n):
    for i in range(n):
        plt.subplot(2,n//2,i+1)
        plt.imshow(items[i])
        plt.axis('off')

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

