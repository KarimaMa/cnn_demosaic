import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils import data
from PIL import Image
from dense_dataset import Dataset
from dense_dense_model import DemosaicCNN
from config import *

def ids_from_file(filename):
    ids = [l.strip() for l in open(filename, "r")]
    return ids

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

BATCH_SIZE = 64
train_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 8}
train_ids = ids_from_file("./dense_train_ids.txt") 
train_dataset = Dataset(train_ids)
train_generator = data.DataLoader(train_dataset, **train_params)

test_params = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 8}
test_ids = ids_from_file("./dense_test_ids.txt")
test_dataset = Dataset(test_ids)
test_generator = data.DataLoader(test_dataset, **test_params)

"""
model parameters
"""
model = DemosaicCNN()
model.to(device)

"""
training parameters
"""
learning_rate = 1e-3
decay = 1e-5
epochs = 70

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=decay)
criterion = nn.MSELoss()

test_losses = []
batch_count = 0
min_counter = 0
min_test_loss = 1e10
eps = 1e-5
B = PATCH_W//2

for epoch in range(epochs):
    total_loss = 0
    for X in train_generator:
        batch_count += 1
        out = model(X["bayer"])
        optimizer.zero_grad()
        loss = criterion(out[...,B:-B,B:-B].to(device), X["target"][...,B:-B,B:-B].to(device))
        total_loss += (loss / (len(train_dataset)/BATCH_SIZE))
        loss.backward()
        optimizer.step()
        if batch_count % 150 == 0:
            print("epoch {} batch loss {}".format(epoch, loss.item()))
            
    print("epoch {} train loss {}".format(epoch, total_loss.item()))

    total_loss = 0
    for X in test_generator:
        with torch.no_grad():
            out = model(X["bayer"])
            loss = criterion(out[...,B:-B,B:-B].to(device), X["target"][...,B:-B,B:-B].to(device))
            total_loss += (loss.item() / (len(test_dataset)/BATCH_SIZE))

    test_losses.append(total_loss)
    print("epoch {} test loss {}".format(epoch, total_loss))
    print("min test loss {}".format(min(test_losses)))
    print("===============")

    if total_loss >= (min_test_loss - eps):
        min_counter += 1
    else:
        min_test_loss = total_loss
        min_counter = 0

    if min_counter > 4 and learning_rate > 1e-5:
        min_counter = 0
        learning_rate /= 10
        min_test_loss = min(test_losses)
        print("decreasing learning rate: {:.7f}".format(learning_rate))

print("test losses")    
print(test_losses)
print("min test loss: {:.5f} at epoch {}".format(min(test_losses)), test_losses.index(min(test_losses)))
print("max test loss: {:.5f} at epoch {}".format(max(test_losses)), test_losses.index(max(test_losses)))
