import ringity.ringscore
from main import MyModInstance

import itertools as it 
import argparse
import ringity
import tqdm
import uuid
import os

def make_and_save_network(output_folder, n_nodes,r,beta,c):
    
    while True:
        try:
            graph_obj = MyModInstance(n_nodes=n_nodes, r=r, beta=beta, c=c)
            folder = os.path.join(output_folder, f"network_{uuid.uuid4()}")
            os.makedirs(folder, exist_ok=True)
            graph_obj.save_info(folder, verbose=True)
            break
        except ringity.utils.exceptions.DisconnectedGraphError:
            print("Network disconnected, retrying...")
        

def main():
    """Generate and save instances of the network model with varying beta and r parameters."""
    parser = argparse.ArgumentParser(description="Generate and save a network.")
    parser.add_argument("--n_nodes", type=int, default=200, help="Number of nodes (default 200)")
    parser.add_argument("--n_networks", type=int, default=50, help="Number of networks to create per parameter config (default 50)")
    parser.add_argument("--c", type=float, default=1.0, help="The parameter c in network construction (default 1.0)")
    parser.add_argument("--o", type=str, default="test_network", help="The output folder")

    args = parser.parse_args()
    
    output_folder = args.o
    print(output_folder)
    
    beta_values = [0.8, 0.85, 0.9, 0.95, 1.0]
    r_values    = [0.1, 0.15, 0.2, 0.25]
    
    param_pairs = list(it.product(r_values, beta_values))
    
    for _, (r, beta) in tqdm.tqdm(it.product(range(args.n_networks), param_pairs), total= args.n_networks * len(param_pairs)):
        
        make_and_save_network(output_folder, args.n_nodes,r,beta,args.c)

#make_and_save_network("test_network", 50, 0.1, 0.9, 1.0)
#main()

beta_values = [0.8, 0.85, 0.9, 0.95, 1.0]
r_values    = [0.1, 0.15, 0.2, 0.25]

param_pairs = list(it.product(r_values, beta_values))

n_reps = 5
for _, (r, beta) in tqdm.tqdm(it.product(range(n_reps), param_pairs), total=5 * len(param_pairs)):
    
    make_and_save_network("test", 30,r,beta,1.0)

