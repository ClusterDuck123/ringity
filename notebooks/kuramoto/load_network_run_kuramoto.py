import ringity.ringscore
from main import MyModInstance

import itertools as it 
import argparse
import ringity
import tqdm
import uuid
import os

def run_and_save(network_folder,T,dt):
    
    graph_obj = MyModInstance.load_instance(network_folder)
    
    run = graph_obj.run(T=T,dt=dt)
    

  
    run.save_run(network_folder, verbose=False)

    
    
def main():
    """Main function to run batch simulations of Kuramoto models on loaded networks."""
    parser = argparse.ArgumentParser(description="Run Kuramoto simulation.")
    parser.add_argument("--i", type=str, default="test_network", help="The output folder")
    parser.add_argument("--T", type=float, default=1000, help="Time to run the system for (not the number of timesteps! that is floor(T/dt))")
    parser.add_argument("--dt", type=float, default=0.001, help="time interval per step")


    args = parser.parse_args()
    
    input_folder  = args.i

    for subfolder in tqdm.tqdm(os.listdir(input_folder)):
        run_and_save(os.path.join(input_folder, subfolder),args.T,args.dt)




main()

