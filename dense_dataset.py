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

def dump(x):
    if len(x.size()) == 3:
        print(x[:,:,:])
    else:
        print(x[:])


class Dataset(data.Dataset):
    def __init__(self, list_IDs):
        self.list_IDs = list_IDs # patch filenames

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        image_f = os.path.join(ID, "im.data")
        bayer_f = os.path.join(ID, "dense_bayer.data")
        target_f = os.path.join(ID, "missing_bayer.data")

        image = np.fromfile(image_f, dtype=np.uint8).reshape((3, IMG_H, IMG_W)).astype(np.float32) / 255
        bayer = np.fromfile(bayer_f, dtype=np.uint8).reshape((4, IMG_H//2, IMG_W//2)).astype(np.float32) / 255
        target = np.fromfile(target_f, dtype=np.uint8).reshape((8, IMG_H//2, IMG_W//2)).astype(np.float32) / 255
        
        image = torch.from_numpy(image).float()
        bayer = torch.from_numpy(bayer).float()
        target = torch.from_numpy(target).float()

        i = random.randrange(0, IMG_W//2 - PATCH_W, 1)
        j = random.randrange(0, IMG_W//2 - PATCH_W, 1)

        image_patch = image[:,i*2:i*2+PATCH_W*2, j*2:j*2+PATCH_W*2]
        bayer_patch = bayer[:,i:i+PATCH_W, j:j+PATCH_W]
        target_patch = target[:,i+PATCH_W//2, j+PATCH_W//2].reshape((8,1,1))

        if debug:
            print('image')
            #dump(image_patch.data)
            print(image_patch[:,4:6,4:6])
            print('bayer')
            dump(bayer_patch.data)
            print('target')
            dump(target_patch.data)
            print("=======")
        """
        target = torch.zeros((1,128,128)).float()
        target[:,:,:] = image[1,:,:] 
        """
        X = { 
            "image": image_patch,
            "bayer": bayer_patch,
            "target" : target_patch,
            "ID": ID
            }

        return X 
