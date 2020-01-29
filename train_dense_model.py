import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils import data
from PIL import Image
from dense_dataset import Dataset
from dense_model import DemosaicCNN

def ids_from_file(filename):
    ids = [l.strip() for l in open(filename, "r")]
    return ids

BATCH_SIZE = 64
train_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 4}
train_ids = ids_from_file("./train_ids.txt") 
train_dataset = Dataset(train_ids)
train_generator = data.DataLoader(train_dataset, **train_params)

test_params = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 4}
test_ids = ids_from_file("./test_ids.txt")
test_dataset = Dataset(test_ids)
test_generator = data.DataLoader(test_dataset, **test_params)

"""
model parameters
"""
k = 3
temp = 0.5
model = DemosaicCNN(k)

"""
training parameters
"""
learning_rate = 1e-3
decay = 1e-5
epochs = 36

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=decay)
criterion = nn.MSELoss()

test_losses = []
batch_count = 0
for epoch in range(epochs):
    total_loss = 0
    for X in train_generator:
        batch_count += 1
        out = model(X["bayer"])
        optimizer.zero_grad()
        loss = criterion(out, X["target"])
        total_loss += (loss / (len(train_dataset)/BATCH_SIZE))
        loss.backward()
        optimizer.step()

    print("epoch {} train loss {}".format(epoch, total_loss.item()))

    total_loss = 0
    for X in test_generator:
        with torch.no_grad():
            out = model(X["bayer"])
            loss = criterion(out, X["target"])
            total_loss += (loss.item() / (len(test_dataset)/BATCH_SIZE))
    print("epoch {} test loss {}".format(epoch, total_loss))
    test_losses.append(total_loss)

    if epoch > 3:
        diff1 = abs(test_losses[-1] - test_losses[-2]) 
        diff2 = abs(test_losses[-2] - test_losses[-3])
        diff3 = abs(test_losses[-3] - test_losses[-4])
        avg_diff = (diff1 + diff2 + diff3) / 3
        if avg_diff < 0.00005:
            learning_rate /= 10
            print("decreasing learning rate: {:.3f}".format(learning_rate))

print("test losses")    
print(test_losses)
print(min(test_losses))
print(test_losses.index(min(test_losses)))
print(max(test_losses))
print(test_losses.index(max(test_losses)))
exit()

# test model
total_loss = 0
for X in test_generator:
    out = model(X["bayer"])
    loss = criterion(out, X["target"])
    total_loss += (loss.item() / (len(test_dataset)/BATCH_SIZE))

for i in range(len(out)):
    print("{} {}".format(out[i], X["target"][i]))

print(len(test_dataset))
print("test loss {:.4f}".format(total_loss))
