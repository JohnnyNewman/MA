# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 19:44:49 2020

@author: Nils
"""



import pandas as pd
import os

import copy
import pickle
import json

if __name__ == "__main__":
    
    
    base_dir = "./"
    
    cwd = os.getcwd()
    os.chdir(base_dir)
    
    data_store = {}
    
    # df_surface_flow_total = pd.DataFrame()
    
    os.makedirs("CollectedData", exist_ok=True)
    
    for dir1 in os.listdir(base_dir):
        if os.path.isdir(dir1) and os.path.isdir(os.path.join(dir1, "DESIGNS")):            
            for dsn_dir in os.listdir(os.path.join(dir1, "DESIGNS")):
                if os.path.isdir(os.path.join(dir1, "DESIGNS")):
                    print(dir1, dsn_dir)
                    
                    try:
                        dsn_data = {}
                        dsn_data["surface_flow"] = pd.read_csv(os.path.join(dir1, "DESIGNS", dsn_dir, "DIRECT", "surface_flow.csv")).to_dict()
                        dsn_data["surface_adjoint_drag"] = pd.read_csv(os.path.join(dir1, "DESIGNS", dsn_dir, "ADJOINT_DRAG", "surface_adjoint.csv")).to_dict()
                        dsn_data["surface_adjoint_cmz"] = pd.read_csv(os.path.join(dir1, "DESIGNS", dsn_dir, "ADJOINT_MOMENT_Z", "surface_adjoint.csv")).to_dict()
                        
                        dsn_data["grad_cd"] = pd.read_csv(os.path.join(dir1, "DESIGNS", dsn_dir, "ADJOINT_DRAG", "of_grad_cd.csv"), quotechar="\"",
                                             names=["VAR", "CD_GRAD", "STEP"], header=0).to_dict()
                        dsn_data["grad_cmz"] = pd.read_csv(os.path.join(dir1, "DESIGNS", dsn_dir, "ADJOINT_MOMENT_Z", "of_grad_cmz.csv"), quotechar="\"",
                                             names=["VAR", "CMZ_GRAD", "STEP"], header=0).to_dict()
                        
                        with open(os.path.join(dir1, "DESIGNS", dsn_dir, "config_DSN.cfg"), "r", encoding="utf-8") as f:
                            dsn_data["config_DSN.cfg"] = f.readlines()
                        with open(os.path.join(dir1, "DESIGNS", dsn_dir, "DIRECT", "log_Direct.out"), "r", encoding="utf-8") as f:
                            dsn_data["log_direct"] = f.readlines()
                        with open(os.path.join(dir1, "DESIGNS", dsn_dir, "flow.meta"), "r", encoding="utf-8") as f:
                            dsn_data["flow.meta"] = f.readlines()
                        
                        
                        
                    except FileNotFoundError as e:
                        print("FileNotFoundError", e)
                    finally:
                        if len(dsn_data):
                            if not dir1 in data_store:
                                data_store[dir1] = {}
                            data_store[dir1][dsn_dir] = copy.deepcopy(dsn_data)
                            
                            fn = os.path.join("CollectedData", dir1, dsn_dir, "dsn_data.json")
                            os.makedirs(os.path.join("CollectedData", dir1, dsn_dir), exist_ok=True)
                            with open(fn, "w") as f:
                                json.dump(data_store, f)
        # except FileNotFoundError as e:
        #     print("FileNotFoundError", e)
    
    os.chdir(cwd)
    
    #pickle.dump(data_store, open("collectedData.p", "wb"))
    # with open("collectedData.json", "w") as f:
    #     #f.write(json.dumps(data_store))
    #     json.dump(data_store, f)
    