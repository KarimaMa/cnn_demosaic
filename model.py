import torch
import torch.nn as nn
import numpy as np


class DemosaicCNN(nn.Module):
    def __init__(self, k):
        super(DemosaicCNN, self).__init__()
        self.k = k

        self.conv1 = torch.nn.Conv2d(1, 16, (3,3), bias=False)
        self.ReLU1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 16, (3,3), bias=False)
        self.ReLU2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(16, 16, (3,3), bias=False)
        self.ReLU3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(16, 16, (3,3), bias=False)
        self.ReLU4 = torch.nn.ReLU()
        self.conv5 = torch.nn.Conv2d(16, 16, (3,3), bias=False)
        self.ReLU5 = torch.nn.ReLU()
        self.conv6 = torch.nn.Conv2d(16, 16, (1,1), bias=False)
        self.ReLU6 = torch.nn.ReLU()
        self.conv7 = torch.nn.Conv2d(16, 1, (1,1), bias=False)
        self.ReLU7 = torch.nn.ReLU()

        """
        self.conv6 = torch.nn.conv2d(16, 16, (3,3), bias=false)
        self.ReLU6 = torch.nn.ReLU()
        self.conv7 = torch.nn.conv2d(16, 1, (3,3), bias=false)
        self.ReLU7 = torch.nn.ReLU()
        self.conv8 = torch.nn.Conv2d(16, 1, (1,1), bias=False)
        self.ReLU8 = torch.nn.ReLU()
        """

        layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]#, self.conv8]
        for layer in layers:
            #torch.nn.init.kaiming_normal_(layer.weight)
            torch.nn.init.xavier_normal_(layer.weight)


    def forward(self, bayer):
        x = self.conv1(bayer)
        x = self.ReLU1(x)
        x = self.conv2(x)
        x = self.ReLU2(x)
        x = self.conv3(x)
        x = self.ReLU3(x)
        x = self.conv4(x)
        x = self.ReLU4(x)
        x = self.conv5(x)
        x = self.ReLU5(x)
        x = self.conv6(x)
        x = self.ReLU6(x)
        x = self.conv7(x)
        x = self.ReLU7(x)

        """

        x = self.conv8(x)
        x = self.ReLU8(x)
        """
        return x



