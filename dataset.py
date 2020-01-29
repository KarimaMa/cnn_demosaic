import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torch.utils import data
import random

IMG_W = 128
IMG_H = 128
PATCH_W = 11

debug = False

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

        bayer = np.fromfile(bayer_f, dtype=np.int16).reshape((1, IMG_W, IMG_H)).astype(np.float32) / 255
        image = np.fromfile(image_f, dtype=np.uint8).reshape((3, IMG_W, IMG_H)).astype(np.float32) / 255

        bayer = torch.from_numpy(bayer).float()
        image = torch.from_numpy(image).float()

        # only pick patches centered on R i is even j is odd
        min_i = PATCH_W//2 
        if min_i % 2 != 0:
            min_i += 1
        min_j = PATCH_W//2
        if min_j % 2 != 1:
            min_j += 1

        i = random.randrange(min_i, IMG_W - PATCH_W//2, 2)
        j = random.randrange(min_j, IMG_W - PATCH_W//2, 2)

        patch = image[:, i-PATCH_W//2:i+PATCH_W//2+1, j-PATCH_W//2:j+PATCH_W//2+1]
        bayer_patch = bayer[:, i-PATCH_W//2:i+PATCH_W//2+1, j-PATCH_W//2:j+PATCH_W//2+1]

        # provide bayer color channels separately too
        # note that this indexing only works for size 5x5 patches
        """
        g = torch.zeros((1, PATCH_W, PATCH_W)).float()
        g[:,0::2,1::2] = bayer_patch[0,0::2,1::2]
        g[:,1::2,0::2] = bayer_patch[0,1::2,0::2]

        r = torch.zeros((1, PATCH_W, PATCH_W)).float()
        r[:,0::2,0::2] = bayer_patch[0,0::2,0::2]

        b = torch.zeros((1, PATCH_W, PATCH_W)).float()
        b[:,1::2,1::2] = bayer_patch[0,1::2,1::2]
        """

        target = torch.zeros((1, 1, 1)).float()
        if self.target == "R":
            target[0,0,0] = patch[0,PATCH_W//2,PATCH_W//2]
        elif self.target == "G":
            target[0,0,0] = patch[1,PATCH_W//2,PATCH_W//2]
        elif self.target == "B":
            target[0,0,0] = patch[2,PATCH_W//2,PATCH_W//2]

        if debug:
            print("bayer patch")
            print(bayer_patch.data)
            """
            print("r")
            print(r.data)
            print("g")
            print(g.data)
            print("b")
            print(b.data)
            """
            print("patch")
            print(patch.data)
            print("target")
            print(target.data)
            print("--------")

        X = { 
            "bayer": bayer_patch,
            "patch": patch,
            "target": target, 
            "ID": ID
            }

        return X 
