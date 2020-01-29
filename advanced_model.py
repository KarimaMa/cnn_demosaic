
import torch
import torch.nn as nn
import numpy as np

class InterpSelector(nn.Module):
    def __init__(self, temp):
        super(InterpSelector, self).__init__()
        self.temp = temp
        k = 5
        self.conv1 = torch.nn.Conv2d(1, 10, (k,k), bias=False, padding=int((k-1)/2))
        self.relu1 = torch.nn.ReLU()
        k = 3
        self.conv2 = torch.nn.Conv2d(10, 10, (k,k), bias=False, padding=int((k-1)/2))
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(10, 3, (1,1), bias=False, padding=0)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.softmax(x/self.temp)
        return x

class LaplacianWeightVert(nn.Module):
    def __init__(self, k):
        super(LaplacianWeightVert, self).__init__()
        self.k = k
        self.g_interp = torch.nn.Conv2d(1, 1, (k, 1), bias=False, padding=(int((k-1)/2),0), stride=1)
        self.rb_interp = torch.nn.Conv2d(2, 1, (k, 1), bias=False, padding=(int((k-1)/2),0), stride=1)

    def forward(self, g, rb):
        g_interp = self.g_interp(g)
        rb_interp = self.rb_interp(rb)
        # only scale down laplacian contribution
        return torch.min(g / (rb_interp + 1e-8), torch.tensor(1.0))


class LaplacianWeightHoriz(nn.Module):
    def __init__(self, k):
        super(LaplacianWeightHoriz, self).__init__()
        self.k = k
        self.g_interp = torch.nn.Conv2d(1, 1, (1, k), bias=False, padding=(0,int((k-1)/2)), stride=1)
        self.rb_interp = torch.nn.Conv2d(2, 1, (1, k), bias=False, padding=(0,int((k-1)/2)), stride=1)

    def forward(self, g, rb):
        g_interp = self.g_interp(g)
        rb_interp = self.rb_interp(rb)
        # only scale down laplacian contribution
        return torch.min(g / (rb_interp + 1e-8), torch.tensor(1.0))

class LaplacianWeight2D(nn.Module):
    def __init__(self, k):
        super(LaplacianWeight2D, self).__init__()
        self.k = 5
        self.g_interp = torch.nn.Conv2d(1, 1, (k, k), bias=False, padding=int((k-1)/2), stride=1)
        self.rb_interp = torch.nn.Conv2d(2, 1, (k, k), bias=False, padding=int((k-1)/2), stride=1)

    def forward(self, g, rb):
        g_interp = self.g_interp(g)
        rb_interp = self.rb_interp(rb)
        # only scale down laplacian contribution
        return torch.min(g / (rb_interp + 1e-8), torch.tensor(1.0))


class GreenInterp(nn.Module):
    def __init__(self, k, temp):
        super(GreenInterp, self).__init__()
        self.k = k
        self.temp = temp
        self.vertical = torch.nn.Conv2d(1, 1, (k, 1), bias=False, padding=(int((k-1)/2),0), stride=1)
        self.chroma_vert = torch.nn.Conv2d(2, 1, (k, 1), bias=False, padding=(int((k-1)/2),0), stride=1)
        self.lweight_vert = LaplacianWeightVert(k)

        self.horizontal = torch.nn.Conv2d(1, 1, (1, k), bias=False, padding=(0,int((k-1)/2)), stride=1)
        self.chroma_horiz = torch.nn.Conv2d(2, 1, (1, k), bias=False, padding=(0,int((k-1)/2)), stride=1)
        self.lweight_horiz = LaplacianWeightHoriz(k)

        self.green_2D = torch.nn.Conv2d(1, 1, (k, k), bias=False, padding=int((k-1)/2), stride=1)
        self.chroma_2D = torch.nn.Conv2d(2, 1, (k, k), bias=False, padding=int((k-1)/2), stride=1)
        self.lweight_2D = LaplacianWeight2D(k)

        self.InterpSelector = InterpSelector(temp)

    def forward(self, g, rb, bayer):
        selection = self.InterpSelector(bayer)

        g_vert = self.vertical(g)
        chroma_vert = self.chroma_vert(rb)
        lweight_vert = self.lweight_vert(g, rb)
        vert = g_vert + lweight_vert * chroma_vert 

        g_horiz = self.horizontal(g)
        chroma_horiz = self.chroma_horiz(rb)
        lweight_horiz = self.lweight_horiz(g, rb)
        horiz = g_horiz + lweight_horiz * chroma_horiz

        g_2D = self.green_2D(g)
        chroma_2D = self.chroma_2D(rb)
        lweight_2D = self.lweight_2D(g, rb)
        aug_2D = g_2D + lweight_2D * chroma_2D


        g_interps = torch.cat((vert, horiz, aug_2D), dim=1)

        weighted_interps = g_interps * selection

        out = torch.sum(weighted_interps, dim=1)
        return out

