from main import MyModInstance,Run
import argparse
import numpy as np
import tqdm
import os


def decide_run_type(run, terminal_length, coherence_threshold, asynchrony_threshold):
    
    terminal_phase_coherence = run.phase_coherence[-terminal_length:]
    
    if np.std(terminal_phase_coherence) > asynchrony_threshold:
        return "asynch"
    
    elif np.mean(terminal_phase_coherence) < 1 - coherence_threshold:
        return "traveling"
    
    else:
        return "coherent"

def main():
    """Generate and save instances of the network model with varying beta and r parameters."""
    parser = argparse.ArgumentParser(description="Generate and save a network.")
    parser.add_argument("--target_number", type=int, default=3, help="Number of runs of each type to select and retry.")
    parser.add_argument("--i", type=str, default="test/", help="The input folder")
    parser.add_argument("--o", type=str, default="rerun_test/", help="The output folder")

    parser.add_argument("--terminal_length",      type=int,   default=100,   help="The number of final timesteps to take when classifying final behaviour.")
    parser.add_argument("--coherence_threshold",  type=float, default=0.5, help="Runs with final mean phase_coherence above 1 - coherence_threshold are coherent.")
    parser.add_argument("--asynchrony_threshold", type=float, default=0.001, help="Runs with final st.dev. in phase_coherence above asynchrony_threshold are asynchronous.")

    args = parser.parse_args()
    
    target_number = args.target_number
    input_folder  = args.i
    output_folder = args.o
    
    terminal_length      = args.terminal_length
    coherence_threshold  = args.coherence_threshold
    asynchrony_threshold = args.asynchrony_threshold
    
    n_networks_saved = {"asynch":0, "traveling":0, "coherent":0}
    
    for network_folder in tqdm.tqdm(os.listdir(input_folder)):

        try:
            folder = os.path.join(input_folder,network_folder)
            network = MyModInstance.load_instance(folder)
            network.folder = network_folder 
            network.fullpath = folder
            
            run_folder = os.listdir(f"{folder}/runs/")[0]
            run_path = os.path.join(folder, "runs", run_folder)
            
            run = Run.load_run(run_path)
            run_type = decide_run_type(run,
                                       terminal_length,
                                       coherence_threshold,
                                       asynchrony_threshold
                                       )
            
            if n_networks_saved[run_type] < target_number:
                
                print(run_type)
                print("running new instance...")
                new_run = network.run(dt=0.1)
                new_run.natfreqs terminal_length= run.natfreqs
                new_run.initialize_simulation()
                

                new_network_folder = os.path.join(output_folder, f"{run_type}_{network.folder}")
                network.save_info(new_network_folder)
                new_run.save_run(new_network_folder,verbose=True)
                
                n_networks_saved[run_type] += 1

        except FileNotFoundError as e:
            
            print(e)
        
    print(n_networks_saved)

if __name__ == "__main__":
    main()