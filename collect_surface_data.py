# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 23:35:36 2020

@author: Nils
"""



import pandas as pd
import os


base_dir = "./"

cwd = os.getcwd()
os.chdir(base_dir)

df_surface_flow_total = pd.DataFrame()

for dir in os.listdir("DESIGNS"):
    print(dir)
    try:
        df_surface_flow_dsn = pd.read_csv(os.path.join("DESIGNS", dir, "DIRECT", "surface_flow.csv"))
    except FileNotFoundError:
        print(dir, "FileNotFoundError")
        continue
    df_surface_flow_dsn["DIR"] = dir
    if df_surface_flow_total.empty:
        df_surface_flow_total = df_surface_flow_dsn.copy()
    else:
        df_surface_flow_total = df_surface_flow_total.append(df_surface_flow_dsn)
    print(df_surface_flow_total.shape)

df_surface_adjoint_drag_total = pd.DataFrame()

for dir in os.listdir("DESIGNS"):
    print(dir)
    try:
        df_surface_adjoint_drag_dsn = pd.read_csv(os.path.join("DESIGNS", dir, "ADJOINT_DRAG", "surface_adjoint.csv"))
    except FileNotFoundError:
        print(dir, "FileNotFoundError")
        continue
    df_surface_adjoint_drag_dsn["DIR"] = dir
    if df_surface_adjoint_drag_total.empty:
        df_surface_adjoint_drag_total = df_surface_adjoint_drag_dsn.copy()
    else:
        df_surface_adjoint_drag_total = df_surface_adjoint_drag_total.append(df_surface_adjoint_drag_dsn)
    print(df_surface_adjoint_drag_total.shape)


os.chdir(cwd)

df_surface_flow_total.to_csv("df_surface_flow_total.csv")
df_surface_adjoint_drag_total.to_csv("df_surface_adjoint_drag_total.csv")