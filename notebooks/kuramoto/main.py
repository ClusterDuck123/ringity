import numpy as np
import networkx as nx
import ringity
import json
import pandas as pd
import os
import uuid
import tqdm
import argparse
import itertools as it
from kuramoto import Kuramoto

class MyModInstance:
    """
    Represents a modular instance of a network with a given number of nodes, connectivity, and parameters.
    Can generate and simulate networks using Kuramoto dynamics.
    """
    def __init__(self, n_nodes=200, r=0.1, beta=0.9, c=1.0, from_scratch=True):
        """
        Initializes the network with given parameters.
        """
        self.n_nodes = n_nodes
        self.r = r
        self.beta = beta
        self.c = c
        self.runs = []

        if from_scratch:
            self.initialize_network()
        else:
            self.adj_mat = None
            self.ring_score = None

    def initialize_network(self):
        """Generates a new network and computes its properties."""
        self.graph_nx, self.positions = ringity.network_model(
            self.n_nodes, c=self.c, r=self.r, beta=self.beta, return_positions=True
        )
        self.ring_score = ringity.ring_score(self.graph_nx)
        self.adj_mat = nx.to_numpy_array(self.graph_nx)
        
    def run(self, dt=0.001, T=1000):
        """Runs one simulation and returns the Run object."""
        return Run(self.adj_mat, dt=dt, T=T)
          
    def run_many(self, n_runs=10, dt=0.001, T=1000):
        """Runs multiple simulations of the network."""
        self.runs = [Run(self.adj_mat, dt=dt, T=T) for _ in range(n_runs)]
    
    def save_json(self, data, path):
        """Helper function to save a dictionary as a JSON file."""
        with open(path, "w") as fp:
            json.dump(data, fp)

    def save_info(self, folder, verbose=True):
        """Saves network parameters and adjacency matrix to a folder."""
        os.makedirs(folder, exist_ok=True)
        os.makedirs(f"{folder}/runs", exist_ok=True)

        params = {
            "n_nodes": self.n_nodes,
            "r": self.r,
            "beta": self.beta,
            "c": self.c,
            "ring_score": self.ring_score,
        }
        self.save_json(params, f"{folder}/network_params.json")

        if verbose:
            pd.DataFrame(self.adj_mat).to_csv(f"{folder}/network.csv")
            pd.Series(self.positions).to_csv(f"{folder}/positions.csv")

    def run_and_save(self, folder, verbose=False,dt=0.001, T=1000):
        """Runs a single simulation and saves the results."""
        run = Run(self.adj_mat,dt=dt, T=T)
        run.save_run(folder, verbose=verbose)

    @staticmethod
    def load_instance(folder):
        """Loads a saved MyModInstance from a folder."""
        with open(f"{folder}/network_params.json", "r") as fp:
            params = json.load(fp)
        
        instance = MyModInstance(
            n_nodes=params["n_nodes"],
            r=params["r"],
            beta=params["beta"],
            c=params["c"],
            from_scratch=False
        )
        instance.ring_score = params["ring_score"]
        instance.adj_mat = pd.read_csv(f"{folder}/network.csv", index_col=0).values
        instance.positions = pd.read_csv(f"{folder}/positions.csv", index_col=0).values.flatten()
        return instance

class Run:
    """Represents a single run of a Kuramoto model simulation on a given network."""
    def __init__(self, adj_mat, dt=0.001, T=1000, angles_vec=None, from_scratch=True):
        """Initializes a Kuramoto simulation."""
        self.n_nodes = adj_mat.shape[0] if adj_mat is not None else 200
        self.T = T
        self.dt = dt
        self.timeframe = np.linspace(0, T, int(T // dt) + 1)

        self.adj_mat = adj_mat
        self.angles_vec = angles_vec 
        
        if from_scratch:
            self.natfreqs = 1 + 0.15 * np.random.randn(self.n_nodes)
            self.initialize_simulation()
    
    def initialize_simulation(self):
        """Sets up and runs the Kuramoto simulation."""
        
        model = Kuramoto(coupling=1.0, dt=self.dt, T=self.T, n_nodes=self.n_nodes, natfreqs=self.natfreqs)
        self.activity = model.run(adj_mat=self.adj_mat, angles_vec=self.angles_vec)
        self.phase_coherence = Kuramoto.phase_coherence(self.activity)
        self.init_conditions = self.activity[:, 0]
    
    def calculate_stats(self, terminal_length=200):
        
        self.terminal_length = terminal_length
        self.terminal_mean = np.mean(self.phase_coherence[-terminal_length:])
        self.terminal_std  = np.std( self.phase_coherence[-terminal_length:])
        
    def save_run(self, folder, verbose=False,verbose_mid = False):
        """Saves simulation results to a specified folder."""
        
        self.calculate_stats()
            
        run_id = str(uuid.uuid4())
        run_folder = f"{folder}/runs/{run_id}"
        os.makedirs(run_folder, exist_ok=True)

        

        
        if verbose_mid:
            pd.DataFrame(self.natfreqs).to_csv(f"{run_folder}/natfreqs.csv")
            pd.DataFrame(self.init_conditions).to_csv(f"{run_folder}/init_conditions.csv")
            pd.DataFrame(self.phase_coherence).to_csv(f"{run_folder}/phase_coherence.csv")
        
        if verbose:
            pd.DataFrame(self.activity).to_csv(f"{run_folder}/full.csv")

        with open(f"{run_folder}/run_stats.json", "w") as fp:
            json.dump({"terminal_length": self.terminal_length,
                       "terminal_std": self.terminal_std,
                       "terminal_mean": self.terminal_mean}, fp)
            
        with open(f"{run_folder}/run_params.json", "w") as fp:
            json.dump({"dt": self.dt, "T": self.T}, fp)

    @staticmethod
    def load_run(run_folder,verbose=False):
        """Loads a saved Run from a folder."""
        with open(f"{run_folder}/run_params.json", "r") as fp:
            params = json.load(fp)
        
        run = Run(None, dt=params["dt"], T=params["T"], from_scratch=False)
        try:
            run.natfreqs = pd.read_csv(f"{run_folder}/natfreqs.csv", index_col=0).values.flatten()
        except FileNotFoundError:
            pass
        
        try:
            run.init_conditions = pd.read_csv(f"{run_folder}/init_conditions.csv", index_col=0).values.flatten()
        except FileNotFoundError:
            pass
                
        try:
            with open(f"{run_folder}/run_stats.json", "r") as fp:
                stats = json.load(fp)
                for i,v in stats.items():
                    run.__setattr__(i,v)
        except FileNotFoundError:
            pass

        try:
            run.phase_coherence = pd.read_csv(f"{run_folder}/phase_coherence.csv", index_col=0).values.flatten()
        except FileNotFoundError:
            pass
        

        if verbose:
            run.activity = pd.read_csv(f"{run_folder}/full.csv", index_col=0).values
            
        return run

def main():
    """Main function to run batch simulations of Kuramoto models with different parameters."""
    parser = argparse.ArgumentParser(description="Run Kuramoto simulation.")
    parser.add_argument("--n_runs", type=int, default=1000, help="Number of runs (default 1000)")
    parser.add_argument("--n_nodes", type=int, default=200, help="Number of nodes (default 200)")
    args = parser.parse_args()
    
    beta_values = [0.8, 0.85, 0.9, 0.95, 1.0]
    r_values = [0.1, 0.15, 0.2, 0.25]
    param_pairs = list(it.product(r_values, beta_values))
    
    for _, (r, beta) in tqdm.tqdm(it.product(range(50), param_pairs), total=50 * len(param_pairs)):
        try:
            graph_obj = MyModInstance(n_nodes=args.n_nodes, r=r, beta=beta, c=1.0)
            folder = f"data/concise/parameter_array/network-{uuid.uuid4()}"
            os.makedirs(folder, exist_ok=True)
            graph_obj.save_info(folder, verbose=False)
            graph_obj.run_and_save(folder)
        except Exception as e:
            print(f"Error in simulation: {e}")

if __name__ == "__main__":
    main()
