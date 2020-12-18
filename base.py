# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 22:07:23 2020

@author: Nils
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import re
from scipy import linalg
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        
        self.hidden_size = 16
        self.bidirectional = True
        
        c1, c2, c3 = 32, 64, 32
        c4, c5, c6 = 32, 64, 64
                
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=2,out_channels=c1,kernel_size=15,stride=1, padding=7),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(in_channels=c1,out_channels=c2,kernel_size=15,stride=1, padding=7),
            nn.PReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=c2,out_channels=c3,kernel_size=15,stride=1, padding=7),
            nn.PReLU(),
            nn.Linear(32,64),
            #nn.PReLU(),
            #nn.Linear(64,32),
            #nn.PReLU(),
            nn.Conv1d(in_channels=c3,out_channels=1,kernel_size=1,stride=1),
        )
    
        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=3),
            nn.Linear(64,64),
            nn.ConvTranspose1d(in_channels=1, out_channels=c4, kernel_size=7,stride=2),
            nn.PReLU(),
            nn.ConvTranspose1d(in_channels=c4, out_channels=c5, kernel_size=7,stride=2),
            nn.PReLU(),
            #nn.MaxPool1d(kernel_size=2),
            nn.ConvTranspose1d(in_channels=c5, out_channels=c6, kernel_size=7,stride=2),
            nn.PReLU(),
            #nn.MaxPool1d(kernel_size=2),
            nn.ConvTranspose1d(in_channels=c6, out_channels=4, kernel_size=7,stride=2),
        )
    
    def forward(self, x):
        
        x_stacked = torch.cat((x, x, x), dim=2)
        
        z = self.encoder(x)
        r = torch.randn_like(z)
        
        y = self.decoder(z + r)
        
        d = y.shape[2] - x.shape[2]
        d0 = int(d/2)
        #print(d)
        
        out_mask = torch.ones(y.shape[2], dtype=torch.long)
        out_mask[:d0] = 0
        out_mask[-d0+d:] = 0
        
        #print(d,d0)
        
        return y[:,:,d0:d0-d]
        #return torch.index_select(y, dim=2, index=out_mask)
    
    def wo_rand(self, x):
        
        x_stacked = torch.cat((x, x, x), dim=2)
        
        z = self.encoder(x)
        #r = torch.randn_like(z)
        
        y = self.decoder(z)
        
        d = y.shape[2] - x.shape[2]
        d0 = int(d/2)
        
        out_mask = torch.ones(y.shape[2], dtype=torch.long)
        out_mask[:d0] = 0
        out_mask[-d0+d:] = 0
        
        #print(d,d0)
        
        return y[:,:,d0:d0-d]
    



if __name__ == "__main__":
    
        
    tm = TestModel()
    
    x = torch.rand(1,2,192)
    
    y = tm(x)
    print(y.shape, count_parameters(tm))
    
    
    
    
    
    