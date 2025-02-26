import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ringity.networkmodel
import ringity.networkmodel.modelparameterbuilder
import ringity.networkmodel.transformations
import seaborn as sns
import matplotlib.pyplot as plt
import json
import sklearn.ensemble
import pandas as pd
import scipy.sparse
from kuramoto import Kuramoto
import uuid
import os

import ringity

class MyModInstance:
    
    def __init__(self,
                n_nodes = 500,
                r = 0.1, 
                beta=0.9,
                density = 0.1
                ):

        self.n_nodes=n_nodes
        self.r=r 
        self.beta=beta
        self.density=density
        
        self.graph_nx,self.positions = ringity.network_model(n_nodes,density=density,r=r,beta=beta,return_positions=True)
        ringity.networkmodel.transformations
        self.ring_score = ringity.ring_score(self.graph_nx)

        # extract adjacency
        self.adj_mat = nx.to_numpy_array(self.graph_nx)

    def run(self,
            n_runs=10,
            dt = 0.01,
            T = 1
            ):
        
        self.runs = [Run(self.adj_mat,
                         dt = dt,
                         T = T) 
                     for _ in range(n_runs)]
    
    def classify_runs(self):
        for i in self.runs:
            i.classify_run()

        
    def save_info(self, folder,verbose=False):
        
        os.makedirs(f"{folder}/",exist_ok=True)
        os.makedirs(f"{folder}/runs",exist_ok=True)
        
        params = {"n_nodes":self.n_nodes,
                  "r":self.r,
                  "beta":self.beta,
                  "density":self.density,
                  "ring_score":self.ring_score}
        
        fp = open(f"{folder}/network_params.json","w")
        json.dump(params,fp)
        fp.close()

        if verbose:
            pd.DataFrame(self.adj_mat).to_csv(f"{folder}/network.csv")
            pd.Series(self.positions).to_csv(f"{folder}/positions.csv")
        
        for run in self.runs:
            
            uuid_ = str(uuid.uuid4())
            os.makedirs(f"{folder}/runs/{uuid_}/",exist_ok=True)
            
            if verbose:
                pd.DataFrame(run.activity).to_csv(f"{folder}/runs/{uuid_}/full.csv")
                pd.DataFrame(run.natfreqs).to_csv(f"{folder}/runs/{uuid_}/natfreqs.csv")
            
            params = {"dt":run.dt,"T":run.T}
            fp = open(f"{folder}/runs/{uuid_}/run_params.json","w")
            json.dump(params,fp)
            fp.close()
            
            try:
                params = {"mean":run.mean,"std":run.std,"terminal_length":run.terminal_length}
                fp = open(f"{folder}/runs/{uuid_}/run_summary.json","w")
                json.dump(params,fp)
                fp.close()
            except AttributeError as e:
                print(e)

class Run:
    
    def __init__(self,
                 adj_mat,
                 dt = 0.01,
                 T = 10,
                 angles_vec = None
                 ):
        
        self.n_nodes = adj_mat.shape[0]
        self.natfreqs = np.random.randn(self.n_nodes)

        self.T = T
        self.dt = dt
        
        # Instantiate model with parameters
        model = Kuramoto(coupling=3, dt=dt, T=T, n_nodes=self.n_nodes,natfreqs=self.natfreqs)
        self.timeframe = np.linspace(0,T,int(T//dt)+1)

        # Run simulation - output is time series for all nodes (node vs time)
        self.activity = model.run(adj_mat=adj_mat,angles_vec=angles_vec) 
        self.phase_coherence = Kuramoto.phase_coherence(self.activity)
    
    def classify_run(self,
                     terminal_length=100
                     ):
        terminal_length = 100
        terminal_activity_values = self.phase_coherence[-terminal_length:]
        self.terminal_length = terminal_length
        self.mean=np.mean(terminal_activity_values)
        self.std = np.std(terminal_activity_values)
        return (self.mean,self.std)

"""
os.makedirs("data/", exist_ok=True)
for i in range(10):
    
    try:
        # randomly generate ringy graph
        graph_obj = MyModInstance(
                    n_nodes = 500,
                    r = 0.3*np.random.rand(), 
                    beta=0.7+0.3*np.random.rand(),
                    density = 0.05*np.random.rand()
                    )
        
        # run parameters
        T = 100000
        dt = T/1000
        
        graph_obj.run(n_runs=20,T=T,dt=dt)
        graph_obj.classify_runs()

        uuid_ = str(uuid.uuid4())
        folder = f"data/verbose/network-{uuid_}"
        os.makedirs(folder,exist_ok=True)
        graph_obj.save_info(folder,verbose=True)
    except Exception as e:
        print(e)
"""

import itertools as it
beta_values = np.linspace(0.7,0.999,11)
r_values = np.linspace(0.05,0.35,11)
param_pairs = list(it.product(r_values,beta_values))
for i in range(100):
    for r,beta in param_pairs:#tqdm.tqdm(param_pairs):
    

        try:
            # randomly generate ringy graph
            graph_obj = MyModInstance(
                        n_nodes = 500,
                        r = r, 
                        beta=beta,
                        density = 0.03
                        )
            
            print("success!",
                r,
                beta
                )
            
            # run parameters
            T = 100000
            dt = T/1000
            
            graph_obj.run(n_runs=1,T=T,dt=dt)
            graph_obj.classify_runs()

            uuid_ = str(uuid.uuid4())
            folder = f"data/concise/parameter_array/network-{uuid_}"
            os.makedirs(folder,exist_ok=True)
            graph_obj.save_info(folder,verbose=False)
            
        except Exception as e:
            #print(e)
            pass


