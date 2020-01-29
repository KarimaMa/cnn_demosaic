import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils import data
from PIL import Image
from advanced_dataset import Dataset
from model import DemosaicCNN
from advanced_model import GreenInterp

def ids_from_file(filename):
    ids = [l.strip() for l in open(filename, "r")]
    return ids

BATCH_SIZE = 32
train_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 4}
train_ids = ids_from_file("./train_ids.txt") 
train_dataset = Dataset(train_ids, "G")
train_generator = data.DataLoader(train_dataset, **train_params)

test_params = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 4}
test_ids = ids_from_file("./test_ids.txt")
test_dataset = Dataset(test_ids, "G")
test_generator = data.DataLoader(test_dataset, **test_params)

"""
model parameters
"""
k = 5
temp = 0.5
model = DemosaicCNN(k)
model = GreenInterp(k, temp)

"""
training parameters
"""
learning_rate = 1e-2
epochs = 10000

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

r_mask = torch.zeros((1,128,128)).float()
r_mask[0, 0::2,0::2] = 1
B = 2 # buffer around border to ignore


num_batches = math.ceil(len(train_dataset)//BATCH_SIZE)
print("num batches {}".format(num_batches))
batch_count = 0
for epoch in range(epochs):
    total_loss = 0
    for X in train_generator:
        batch_count += 1
        rb = torch.cat((X["r"], X["b"]), dim=1)
        out = model(X["g"], rb, X["bayer"])

        optimizer.zero_grad()

        loss = criterion(out[...,B:-B,B:-B], X["target"][...,B:-B,B:-B])
        total_loss += (loss.item() / num_batches)
        loss.backward()
        optimizer.step()

        if batch_count % 20 == 0:
            print("epoch {} batch loss {:.3f}".format(epoch, loss))
    
    torch.save(model.state_dict(), "model.data")
    print("epoch {} batch_count {} loss {}".format(epoch, batch_count, total_loss))

# test model
errors = 0
count = 0
for X in test_generator:
    out = model(X["bayer"], X["r"], X["g"], X["b"])
    loss = criterion(out[...,B:-B,B:-B], X["target"][...,B:-B,B:-B])
    total_loss += (loss / (len(test_dataset)/BATCH_SIZE))

print(len(test_dataset))
print("test loss {:.3f}".format(total_loss))
