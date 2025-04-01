import ringity.ringscore
from main import MyModInstance

import itertools as it 
import argparse
import ringity
import tqdm
import uuid
import os

def run_and_save(network_folder):
    
    graph_obj = MyModInstance.load_instance(network_folder)
    
    run = graph_obj.run()
    

  
    run.save_run(network_folder, verbose=False)

    
    
def main():
    """Main function to run batch simulations of Kuramoto models on loaded networks."""
    parser = argparse.ArgumentParser(description="Run Kuramoto simulation.")
    parser.add_argument("--i", type=str, default="test_network", help="The output folder")
    args = parser.parse_args()
    
    input_folder  = args.i
    for subfolder in tqdm.tqdm(os.listdir(input_folder)):
        run_and_save(os.path.join(input_folder, subfolder))




main()

