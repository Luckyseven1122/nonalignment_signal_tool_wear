import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import adaptive_max_pool1d
from pytorch_tool_wear.spp_layer import spatial_pyramid_pool

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes,kernel_size=3,bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = nn.Conv1d(inplanes, planes,kernel_size=3,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv1d(inplanes, planes,kernel_size=3,bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class resBlock(nn.Module):

    def __init__(self,input_dim,output_dim,filter_number,kernel_size=3,dropout_rate=0.5,stride=1):
        super(resBlock,self).__init__()
        k1 = filter_number
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(input_dim,k1,kernel_size=kernel_size,bias=False)
        self.bn2 = nn.BatchNorm1d(k1)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(k1,output_dim,kernel_size=kernel_size,stride=stride)

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out += residual
        out = self.relu1(out)

        return out



class ResNetWithROI(nn.Module):

    def __init__(self,block_depth):
        super(ResNetWithROI,self).__init__()
        self.BLOCK_DEPTH = block_depth
        self.conv1 = nn.Conv1d(7,64,kernel_size=3,bias=False)
        self.bn1 = nn.BatchNorm1d(7)
        self.relu = nn.ReLU(inplace=True)



    def _block_layer(self,input_dim,output_dim):
        layer = []
        layer.append(resBlock(input_dim,output_dim))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for block_cnt in range(self.BLOCK_DEPTH):
            x = self._block_layer(7,64)

        x = spatial_pyramid_pool(x,1,[int(x.size(1))],64)
        x = nn.Linear(x.size,out_features=3)(x)
        return x