from main import MyModInstance,Run
import tqdm
import numpy as np
import os
import pandas as pd

def load_and_sort_networks_by_parameter(top_folder, beta_centers,r_centers):
    # load all the summaries networks and their runs
    # load them into a dict keyed by the network's parameter values
    
    beta_bin_starts = (beta_centers[:-1]+beta_centers[1:])/2
    r_bin_starts    = (r_centers[:-1]+r_centers[1:])/2
    

    out = {}
    for subfolder in tqdm.tqdm(os.listdir(top_folder)):
        folder = os.path.join(top_folder,subfolder)
        try:
            network = MyModInstance.load_instance(folder)
            network.folder = folder
            
        
            i = np.digitize(network.beta, beta_bin_starts)
            j = np.digitize(network.r, r_bin_starts)
            
            
            try:
                out[i,j].append(network)
            except KeyError:
                out[i,j] = [network]
            
        except FileNotFoundError as e:
            pass
            
    return out



def runs_from_networks(networks):
    
    out = []
    for network in networks:
        run_folder = os.path.join(network.folder, "runs/")
        out.extend(load_runs_from_folder(run_folder))
    
    return out

def load_runs_from_folder(folder):
    
    out = []
    for subfolder in os.listdir(folder):
        full_subfolder_path = os.path.join(folder, subfolder)
        try:
            run = Run.load_run(full_subfolder_path)
            run.folder = full_subfolder_path
            out.append(run)
        except FileNotFoundError as e:
            pass
            


    return out


out=[]
top_folder = "test/"
for subfolder in tqdm.tqdm(os.listdir(top_folder)):
    folder = os.path.join(top_folder,subfolder)
    try:
        network = MyModInstance.load_instance(folder)
        network.folder = folder
        run_folder = os.path.join(network.folder, "runs/")
        runs = load_runs_from_folder(run_folder)

        for run in runs:
            out.append([network.folder, run.folder, network.n_nodes, network.beta, network.c, network.r,network.ring_score, run.terminal_mean,run.terminal_std, run.terminal_length])
    except FileNotFoundError as e:
        pass
       
df = pd.DataFrame(out,columns=["network_folder", "run_folder", "n_nodes", "beta", "c", "r","ring_score", "run_terminal_mean","run_terminal_std", "run_terminal_length"])
df.to_csv("test.csv")

if False:
    
    
    beta_centers,r_centers = np.linspace(0.8,1.0,5),np.linspace(0.1,0.25,4)
    n_beta_vals,n_r_vals = len(beta_centers),len(r_centers)

    parameter_network_dict = load_and_sort_networks_by_parameter(top_folder,beta_centers,r_centers)

    for parameter_index_pair, networks in parameter_network_dict.items():
        
        i,j = parameter_index_pair
        runs = runs_from_networks(networks)
    
