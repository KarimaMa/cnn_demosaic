import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils import data
import random

IMG_W = 128
IMG_H = 128

debug = False

def dump(x):
    if len(x.size()) == 3:
        print(x[:,0:5,0:5])
    else:
        print(x[0:5,0:5])

class Dataset(data.Dataset):
    def __init__(self, list_IDs, target):
        self.list_IDs = list_IDs # patch filenames
        self.target = target

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        bayer_f   = os.path.join(ID, "bayer.data")
        image_f   = os.path.join(ID, "image.data")

        bayer = np.fromfile(bayer_f, dtype=np.int16).reshape((1, IMG_W, IMG_H)).astype(np.float32)
        image = np.fromfile(image_f, dtype=np.uint8).reshape((3, IMG_W, IMG_H)).astype(np.float32) 

        bayer = torch.from_numpy(bayer).float()
        image = torch.from_numpy(image).float()

        # provide bayer color channels separately too
        g = torch.zeros((1, IMG_W, IMG_W)).float()
        g[:,0::2,0::2] = bayer[0,0::2,0::2]
        g[:,1::2,1::2] = bayer[0,1::2,1::2]

        r = torch.zeros((1, IMG_W, IMG_W)).float()
        r[:,0::2,1::2] = bayer[0,0::2,1::2]

        b = torch.zeros((1, IMG_W, IMG_W)).float()
        b[:,1::2,0::2] = bayer[0,1::2,0::2]

        target = torch.zeros((IMG_W, IMG_W)).float()
        if self.target == "R":
            print("not implemented")
            exit()
        elif self.target == "G":
            target[0::2,1::2] = image[1,0::2,1::2]
            target[1::2,0::2] = image[1,1::2,0::2]
        elif self.target == "B":
            print("not implemented")
            exit()

        if debug:
            print("bayer patch")
            dump(bayer.data)
            print("r")
            dump(r.data)
            print("g")
            dump(g.data)
            print("b")
            dump(b.data)
            print("image")
            dump(image.data)
            print("target")
            dump(target.data)
            print("--------")

        X = { 
            "bayer": bayer,
            "image": image,
            "r": r,
            "g": g,
            "b": b,
            "target": target, 
            "ID": ID
            }

        return X 
