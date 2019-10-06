import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import layers as complex
import math
    
class ManifoldNetRes(nn.Module):
    def __init__(self):
        super(ManifoldNetRes, self).__init__()
        self.complex_conv1 = complex.ComplexConv2Deffgroup(1, 20, (5, 5), (2, 2))
        self.complex_conv2 = complex.ComplexConv2Deffgroup(20, 20, (5, 5), (2, 2))
        self.proj2 = complex.manifoldReLUv2angle(20) 
        self.complex_res = complex.ResidualLayer(20, 20, 20, (5, 5), (2, 2))
        self.relu = nn.ReLU()
        self.linear_1 = complex.ComplexLinearangle2Dmw_outfield(20*22*22)
        self.conv_1 = nn.Conv2d(20, 30, (5, 5))
        self.mp_1 = nn.MaxPool2d((2, 2))
        self.conv_2 = nn.Conv2d(40, 50, (5, 5), (3, 3))
        self.bn_1 = nn.BatchNorm2d(30)
        self.bn_2 = nn.BatchNorm2d(50)
        self.mp_2 = nn.MaxPool2d((3, 3))
        self.conv_3 = nn.Conv2d(60, 70, (2, 2))
        self.bn_3 = nn.BatchNorm2d(70)
        self.linear_2 = nn.Linear(70, 30)
        self.linear_4 = nn.Linear(30, 11)
        self.res1=nn.Sequential(*self.make_res_block(30, 40))
        self.id1 = nn.Conv2d(30, 40, (1, 1))
        self.res2=nn.Sequential(*self.make_res_block(50, 60))
        self.id2 = nn.Conv2d(50, 60, (1, 1))
        
        
    def make_res_block(self, in_channel, out_channel):    
        res_block = []
        res_block.append(nn.BatchNorm2d(in_channel))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(in_channel, int(out_channel / 4), (1, 1), bias=False))
        res_block.append(nn.BatchNorm2d(int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4), int(out_channel / 4), (3, 3), bias=False, padding=1))
        res_block.append(nn.BatchNorm2d(int(out_channel / 4)))
        res_block.append(nn.ReLU())
        res_block.append(nn.Conv2d(int(out_channel / 4), out_channel, (1, 1), bias=False))
        return res_block
       
        
    def forward(self, x):
        x = self.complex_conv1(x)
        conv1_x = self.proj2(x)
        x = self.complex_conv2(conv1_x)
        conv2_x = self.proj2(x)
        x = self.complex_res(conv1_x, conv2_x)
        x = self.proj2(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.conv_1(x)
        x = self.bn_1(x)
        x_res = self.relu(x)
        x = self.id1(x_res) + self.res1(x_res)
        x = self.mp_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x_res = self.relu(x)
        x = self.id2(x_res) + self.res2(x_res)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_4(x)
        return x

