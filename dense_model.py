import torch
import torch.nn as nn
import numpy as np


class DemosaicCNN(nn.Module):
    def __init__(self, k):
        super(DemosaicCNN, self).__init__()
        self.k = k

        self.conv1 = torch.nn.Conv2d(4, 16, (3,3), bias=False)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 16, (3,3), bias=False)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(16, 16, (1,1), bias=False)
        self.relu3 = torch.nn.ReLU()
        self.conv4 = torch.nn.Conv2d(16, 16, (1,1), bias=False)
        self.relu4 = torch.nn.ReLU()
        self.conv5 = torch.nn.Conv2d(16, 8, (1,1), bias=False)
        self.relu5 = torch.nn.ReLU()


        """
        self.conv6 = torch.nn.Conv2d(16, 8, (1,1), bias=False)
        self.relu6 = torch.nn.ReLU()
        self.conv7 = torch.nn.Conv2d(16, 16, (3,3), bias=False)
        self.relu7 = torch.nn.ReLU()
        self.conv8 = torch.nn.Conv2d(16, 16, (3,3), bias=False)
        self.relu8 = torch.nn.ReLU()
        self.conv9 = torch.nn.Conv2d(16, 16, (3,3), bias=False)
        self.relu9 = torch.nn.ReLU()
        self.conv10 = torch.nn.Conv2d(16, 8, (3,3), bias=False)
        self.relu10 = torch.nn.ReLU()
        """

        layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]#, self.conv6]#, self.conv7, self.conv8, self.conv9, self.conv10]
        for layer in layers:
            torch.nn.init.xavier_normal_(layer.weight)


    def forward(self, bayer):
        x = self.conv1(bayer)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)

        """
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.conv9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.relu10(x)
        """
        return x



