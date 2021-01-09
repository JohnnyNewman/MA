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

import torch
from torch.nn import functional as F
#from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader

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
    

def save_collected_data():
    dir_filter = ["T007"]
    
    collectedData = {}
    df_grad_cd_list = []
    for dir1 in os.listdir("CollectedData"):
        if os.path.isdir(os.path.join("CollectedData", dir1)):
            if dir1 in dir_filter:
                print(dir1)
                collectedData[dir1] = {}
                for dsn_dir in os.listdir(os.path.join("CollectedData", dir1)):
                    if os.path.isdir(os.path.join("CollectedData", dir1, dsn_dir)):
                        data = json.load(open(os.path.join("CollectedData", dir1, dsn_dir, "dsn_data.json"), "r"))
                        collectedData[dir1][dsn_dir] = data
    
    re_log_line = r"\|\s+\d+(\|\s+-?\d+\.\d+){4}\|\n"
    
    dir_list = []
    dfs = []
    dv_mat = []
    surf_dfs = []
    surf_cd_dfs = []
    
    #df_meta = pd.DataFrame(columns=["AOA"])
    meta_rows = []
    
    result_rows = []
    for dir1 in sorted(collectedData.keys()):
        if dir1 in dir_filter: # or dir1.startswith("T005"):
            for dsn_dir in sorted(collectedData[dir1].keys()):
                dir_str = f"{dir1} {dsn_dir}"
                print(dir_str)
                field_names = ["config_DSN.cfg", "grad_cd", "surface_adjoint_drag", "surface_flow", "log_direct"]
                if len(collectedData[dir1][dsn_dir].keys() & set(field_names)) < len(field_names):
                    print("skipping", dir_str)
                    continue
                if "config_DSN.cfg" in collectedData[dir1][dsn_dir]:
                    config_lines = collectedData[dir1][dsn_dir]["config_DSN.cfg"]
                    dv_line = [l for l in config_lines if l.startswith("DV_VALUE=")][-1].strip()
                    #print(dv_line)
                    dvs = [float(x) for x in dv_line[10:].split(", ")]
                    #if len(dvs) < 30:
                    #    print(dir1, dsn_dir)
                    #    continue
                    #print(dvs)
                    if len(dvs) == 38:
                        dv_mat.append(dvs)
                if "grad_cd" in collectedData[dir1][dsn_dir]:
                    df = pd.DataFrame.from_dict(collectedData[dir1][dsn_dir]["grad_cd"])
                    df["DIR"] = dir_str
                    dfs.append(df)
                    dir_list.append(f"{dir1} {dsn_dir}")
                if "surface_adjoint_drag" in collectedData[dir1][dsn_dir]:
                    surf_df = pd.DataFrame.from_dict(collectedData[dir1][dsn_dir]["surface_adjoint_drag"])
                    surf_df["DIR"] = dir_str
                    surf_cd_dfs.append(surf_df)
                if "surface_flow" in collectedData[dir1][dsn_dir]:
                    surf_df = pd.DataFrame.from_dict(collectedData[dir1][dsn_dir]["surface_flow"])
                    surf_df["DIR"] = dir_str
                    surf_dfs.append(surf_df)
                if "log_direct" in collectedData[dir1][dsn_dir]:
                    log_lines = collectedData[dir1][dsn_dir]["log_direct"]
                    log_lines_iter = [l for l in log_lines if re.match(re_log_line, l)]#[-100:]
                    lls = log_lines_iter[-1].split("|")[1:-1]
                    iteration = int(lls[0])
                    rms_rho = float(lls[1])
                    rms_nu = float(lls[2])
                    c_L = float(lls[3])
                    c_D = float(lls[4])
                    #print(dir_str, c_L, c_D)
                    result_rows.append((dir_str, c_L, c_D, rms_rho, rms_nu))
                # if "flow.meta" in collectedData[dir1][dsn_dir]:
                #     #print(collectedData[dir1][dsn_dir]["flow.meta"])
                #     #break
                #     for l in collectedData[dir1][dsn_dir]["flow.meta"]:
                #         if l.startswith("AOA="):
                #             meta_rows.append((dir_str, float(l.strip().split()[1])))
    
    #dv_mat = np.asarray(dv_mat)
    df_result = pd.DataFrame(columns=["DIR", "c_L", "c_D", "rms_rho", "rms_nu"], data=result_rows)
    #df_meta =  pd.DataFrame(columns=["DIR", "AOA"], data=meta_rows)
    #df_result = df_result.merge(df_meta, on="DIR")
    
    df_surface = pd.concat(surf_dfs)
    df_surface_adjoint_drag_total = pd.concat(surf_cd_dfs)
    df_surface = df_surface.merge(df_surface_adjoint_drag_total, on=["PointID", "DIR"], suffixes=["", "_right"])
    
    return df_result, df_surface


# class AirfoilModel(LightningModule):
    
#     def __init__(self, train_dataset, test_dataset):
#         super().__init__()
        
#         self.batch_size = 128
#         #self.hparams.batch_size = 64
        
#         self.train_dataset = train_dataset
#         self.test_dataset = test_dataset
        
#         c1 = 8
#         c2 = 8
#         k2 = 8
#         c3 = 8 

#         # mnist images are (1, 28, 28) (channels, width, height)
#         self.layer_0 = nn.Conv1d(in_channels=2,out_channels=c1,kernel_size=3,stride=1, padding=1)

#         self.layer_11 = nn.Conv1d(in_channels=c1,out_channels=c2,kernel_size=3,stride=1, padding=1)
#         self.layer_12 = nn.Conv1d(in_channels=c1,out_channels=c2,kernel_size=3,stride=1, padding=1)
#         self.layer_13 = nn.Conv1d(in_channels=c1,out_channels=c2,kernel_size=3,stride=1, padding=1)
#         self.layer_14 = nn.Conv1d(in_channels=c1,out_channels=c2,kernel_size=3,stride=1, padding=1)
#         self.relu_11 = nn.ReLU()
#         self.relu_12 = nn.ReLU()
#         self.relu_13 = nn.ReLU()
#         self.relu_14 = nn.ReLU()

#         self.layer_2 = nn.Conv1d(in_channels=8*4,out_channels=c3,kernel_size=5,stride=2, padding=1)
#         self.relu_2 = nn.ReLU()
        
#         self.layer_3 = torch.nn.Linear(287, 8)
        
        
#         self.layer_5 = torch.nn.Linear(66, 32)
        
#         self.layer_6 = torch.nn.Linear(32, 32)
#         self.relu_6 = nn.ReLU()
#         self.layer_7 = torch.nn.Linear(32, 32)
#         self.relu_7 = nn.ReLU()
#         self.layer_8 = torch.nn.Linear(32, 32)
#         self.relu_8 = nn.ReLU()
#         self.layer_9 = torch.nn.Linear(32, 32)
#         self.relu_9 = nn.ReLU()
#         self.layer_10 = torch.nn.Linear(32, 64)
#         self.relu_10 = nn.ReLU()
        
#         c4 = 32
#         c5 = 16
#         c6 = 8
#         self.deconv_11 = nn.ConvTranspose1d(in_channels=64, out_channels=c4, kernel_size=7, stride=3, dilation=3)
#         self.deconv_12 = nn.ConvTranspose1d(in_channels=c4, out_channels=c5, kernel_size=7, stride=3, dilation=3)
#         self.relu_12 = nn.ReLU()
#         self.deconv_13 = nn.ConvTranspose1d(in_channels=c5, out_channels=c6, kernel_size=7, stride=3, dilation=3)
#         self.relu_13 = nn.ReLU()
#         self.deconv_14 = nn.ConvTranspose1d(in_channels=c6, out_channels=4, kernel_size=7, stride=1, dilation=1)
        
        
#         self.loss_crit = nn.MSELoss()

#     def forward(self, x, Ma, AOA):
#         #batch_size, channels, width, height = x.size()
        
#         ### Encoder
        
#         xxx = torch.cat((x, x, x), dim=2)
#         x_0 = self.layer_0(xxx)

#         x_11 = self.relu_11(self.layer_11(x_0))
#         x_12 = self.relu_12(self.layer_12(x_0))
#         x_13 = self.relu_13(self.layer_13(x_0))
#         x_14 = self.relu_14(self.layer_14(x_0))

#         x_1 = torch.cat((x_11, x_12, x_13, x_14), dim=1)
        
#         x_2 = self.relu_2(self.layer_2(x_1))
#         x_3 = self.layer_3(x_2).view(-1, 1, 64)
        
#         #print(x_3.shape, AOA.shape, Ma.shape)
        
#         ### Decoder
        
#         x_4 = torch.cat((x_3, AOA.view(-1, 1, 1), Ma.view(-1, 1, 1)), dim=2)
        
#         x_5 = self.layer_5(x_4)
        
#         x_6 = self.relu_6(self.layer_6(x_5))
#         x_7 = self.relu_7(self.layer_7(x_6))
        
#         x_8 = self.relu_8(self.layer_8(x_5+x_7))
#         x_9 = self.relu_9(self.layer_9(x_8))
        
#         x_10 = self.relu_10(self.layer_10(x_9+x_7))
        
#         x_11 = self.deconv_11(x_10.view(-1, 64, 1))
#         x_12 = self.relu_12(self.deconv_12(x_11))
#         x_13 = self.relu_13(self.deconv_13(x_12))
#         x_14 = self.deconv_14(x_13)
        
#         y = x_14
        
#         d = y.shape[2] - x.shape[2]
#         d0 = int(d/2)
#         #print(d)
        
#         out_mask = torch.ones(y.shape[2], dtype=torch.long)
#         out_mask[:d0] = 0
#         out_mask[-d0+d:] = 0
        
#         #print(d,d0)
        
#         return y[:,:,d0:d0-d]

    
#     def training_step(self, batch, batch_idx):
#         ma, aoa, dv, x, y = batch
#         x_hat = self(x, ma, aoa)
#         loss = F.mse_loss(x_hat, y)
#         # Logging to TensorBoard by default
#         self.log('train_loss', loss)
#         return loss
    
#     def validation_step(self, batch, batch_idx):
#         ma, aoa, dv, x, y = batch
#         x_hat = self(x, ma, aoa)
#         loss = F.mse_loss(x_hat, y)
#         # Logging to TensorBoard by default
#         self.log('val_loss', loss)
#         return loss
        
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer
    
#     def train_dataloader(self):
#         #print("get dataloader ", self.batch_size)
#         return DataLoader(self.train_dataset, batch_size=self.batch_size)    
    
#     def val_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=len(test_dataset))

class AirfoilModel(LightningModule):
    
    def __init__(self):
        super().__init__()
        
        self.batch_size = 128
        #self.hparams.batch_size = 64
        self.lr=1e-3
        
        self.train_variance = 1.0
        
        c1 = 16
        c2 = 8
        k2 = 8
        c3 = 8 

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_0 = nn.Conv1d(in_channels=2,out_channels=c1,kernel_size=3,stride=1, padding=1)

        self.layer_11 = nn.Conv1d(in_channels=c1,out_channels=c2,kernel_size=3,stride=1, padding=1)
        self.layer_12 = nn.Conv1d(in_channels=c1,out_channels=c2,kernel_size=3,stride=1, padding=1)
        self.layer_13 = nn.Conv1d(in_channels=c1,out_channels=c2,kernel_size=3,stride=1, padding=1)
        self.layer_14 = nn.Conv1d(in_channels=c1,out_channels=c2,kernel_size=3,stride=1, padding=1)
        self.relu_11 = nn.ReLU()
        self.relu_12 = nn.ReLU()
        self.relu_13 = nn.ReLU()
        self.relu_14 = nn.ReLU()

        self.layer_2 = nn.Conv1d(in_channels=8*4,out_channels=c3,kernel_size=5,stride=2, padding=1)
        self.relu_2 = nn.ReLU()
        
        self.layer_3 = torch.nn.Linear(287, 8)
        
        
        c_res = 32
        
        self.layer_5 = torch.nn.Linear(66, c_res)
        
        self.layer_6 = torch.nn.Linear(c_res, c_res)
        self.relu_6 = nn.ReLU()
        self.layer_7 = torch.nn.Linear(c_res, c_res)
        self.relu_7 = nn.ReLU()
        self.layer_8 = torch.nn.Linear(c_res, c_res)
        self.relu_8 = nn.ReLU()
        self.layer_9 = torch.nn.Linear(c_res, c_res)
        self.relu_9 = nn.ReLU()
        self.layer_10 = torch.nn.Linear(c_res, 64)
        self.relu_10 = nn.ReLU()
        
        c4 = 32
        c5 = 16
        c6 = 8
        self.deconv_11 = nn.ConvTranspose1d(in_channels=64, out_channels=c4, kernel_size=7, stride=3, dilation=3)
        self.deconv_12 = nn.ConvTranspose1d(in_channels=c4, out_channels=c5, kernel_size=7, stride=3, dilation=3)
        self.relu_12 = nn.ReLU()
        self.deconv_13 = nn.ConvTranspose1d(in_channels=c5, out_channels=c6, kernel_size=7, stride=3, dilation=3)
        self.relu_13 = nn.ReLU()
        self.deconv_14 = nn.ConvTranspose1d(in_channels=c6, out_channels=1, kernel_size=7, stride=1, dilation=1)
        
        
        self.loss_crit = nn.MSELoss()
    
    
    def airfoil_encoder(self, x):
        
        xxx = torch.cat((x, x, x), dim=2)
        x_0 = self.layer_0(xxx)

        x_11 = self.relu_11(self.layer_11(x_0))
        x_12 = self.relu_12(self.layer_12(x_0))
        x_13 = self.relu_13(self.layer_13(x_0))
        x_14 = self.relu_14(self.layer_14(x_0))

        x_1 = torch.cat((x_11, x_12, x_13, x_14), dim=1)
        
        x_2 = self.relu_2(self.layer_2(x_1))
        x_3 = self.layer_3(x_2).view(-1, 1, 64)
        
        return x_3
    
    def airfoil_predictor(self, x_3, Ma, AOA):
        
        x_4 = torch.cat((x_3, AOA.view(-1, 1, 1), Ma.view(-1, 1, 1)), dim=2)
        
        x_5 = self.layer_5(x_4)
        
        x_6 = self.relu_6(self.layer_6(x_5))
        x_7 = self.relu_7(self.layer_7(x_6))
        
        x_8 = self.relu_8(self.layer_8(x_5+x_7))
        x_9 = self.relu_9(self.layer_9(x_8))
        
        x_10 = self.relu_10(self.layer_10(x_9+x_7))
        
        x_11 = self.deconv_11(x_10.view(-1, 64, 1))
        x_12 = self.relu_12(self.deconv_12(x_11))
        x_13 = self.relu_13(self.deconv_13(x_12))
        x_14 = self.deconv_14(x_13)
        
        y = x_14
        
        d = y.shape[2] - x.shape[2]
        d0 = int(d/2)
        #print(d)
        
        out_mask = torch.ones(y.shape[2], dtype=torch.long)
        out_mask[:d0] = 0
        out_mask[-d0+d:] = 0
        
        #print(d,d0)
        
        return y[:,:,d0:d0-d]
    
    def forward(self, x, Ma, AOA, variance=1.0):
        #batch_size, channels, width, height = x.size()
        
        ### Encoder
        
        x_3 = self.airfoil_encoder(x)
        
        #print(x_3.shape, AOA.shape, Ma.shape)
        
        r = torch.randn_like(x_3, device=self.device)
        x_3 = x_3 + r*variance
        
        ### Decoder
        
        y = self.airfoil_predictor(x_3, Ma, AOA)
        
        return y

    
    def training_step(self, batch, batch_idx):
        ma, aoa, dv, x, y = batch
        #x_hat = self(x, ma, aoa)
        x_3 = self.airfoil_encoder(x)
        r = torch.randn_like(x_3, device=self.device)
        x_3 = x_3 + r*self.train_variance
        x_hat = self.airfoil_predictor(x_3, ma, aoa)
        
        #loss = F.mse_loss(x_hat, y[:,:2,:])
        loss = F.mse_loss(x_hat, y[:,0,:].view(-1,1,192))
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ma, aoa, dv, x, y = batch
        x_hat = self(x, ma, aoa, 0.)
        #loss = F.mse_loss(x_hat, y[:,:2,:])
        loss = F.mse_loss(x_hat, y[:,0,:].view(-1,1,192))
        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def train_dataloader(self):
        #print("get dataloader ", self.batch_size)
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True, pin_memory=True)    
    
    def val_dataloader(self):
        return DataLoader(test_dataset, batch_size=len(test_dataset), num_workers=0, pin_memory=True)
    

if __name__ == "__main__":
    
        
    tm = TestModel()
    
    x = torch.rand(1,2,192)
    
    y = tm(x)
    print(y.shape, count_parameters(tm))
    
    
    
    
    
    