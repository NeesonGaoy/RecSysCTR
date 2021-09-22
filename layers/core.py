# coding:utf-8
'''
Author: sysu.gaoyong
Email: ygaoneeson@gmail.com
Date: 2021-09-18 14:53:54
LastEditTime: 2021-09-22 09:21:34
LastEditors: sysu.gaoyong
FilePath: /RecSysCTR/layers/core.py
Copyright (c) 2011 Neeson.GaoYong All rights reserved.
'''

import os
import numpy as np
import torch
import torch.nn.functional as F


class FM(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super(FM, self).__init__()
        self.reduce_sum = reduce_sum
    def forward(self, id_emb):
        square_of_sum = torch.sum(id_emb, dim=1) ** 2
        sum_of_square = torch.sum(id_emb ** 2, dim=1)
        out = square_of_sum - sum_of_square
        if self.reduce_sum:
            out = out.sum(1, True)
        return 0.5 * out


class CrossNet(torch.nn.Module):
    def __init__(self, input_dim, num_layer):
        super(CrossNet, self).__init__()
        self.input_dim = input_dim
        self.num_layer = num_layer
        self.w = torch.nn.ModuleList([torch.nn.Linear(input_dim, 1, bias=False) for _ in range(self.num_layer)])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(input_dim,)) for _ in range(self.num_layer)])
    def forward(self, x):   
        x0 = x
        for i in range(self.num_layer):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class MLP(torch.nn.Module):
    def __init__(self, input_dim, mlp_dims, dropout):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout
        layers = list()
        for out_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, out_dim))
            # layers.append(torch.nn.BatchNorm1d(out_dim)) # trans to tf pb for inference 
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = out_dim
        self.mlp = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.mlp(x)



