
Here's how this set of scripts works:

There is a central module main.py which contains the key objects and the





An initial script `make_networks.py`, creates 50 different networks for each combination of initial conditions. If the parameter set creates a disconnected network, it retries until 50 are found.
runs

A second script `load_network_run_kuramoto.py`, is run 1000 times in parallel for each network, each time loading up the network and running anew version of the with different natural frequencies and initial conditions. A summary of each run is then saved, minimally the
- phase coherence
-






A helper script, `load_network_run_kuramoto.sbatch` manages the paralellized workload on slurm, with the correct time and memory requirements.


specific runs, showing particular behaviours are selected, re-ran, and saved with more detail in `select_and_rerun_interesting.py` and visualised in more detail in `detailed_run_visualisation.py. <-TODO



The run summaries are then loaded to create dotplot in  `parameter_array_dotplot.py`


`ring_score_vs_dynamics_scatterplot.py` <-TODO












## Using the `main` Module Directly

### Loading Existing Data

- `MyModInstance.load_instance(folder)`:
  Loads a previously saved network from a specified folder, restoring its key properties, such as the adjacency matrix and ring score.

- `Run.load_run(run_folder)`:
  Loads a previously saved simulation run, including the phase coherence, natural frequencies, and initial conditions.

### Creating and Simulating Networks

- `MyModInstance(n_nodes=200, r=0.1, beta=0.9, c=1.0)`:
  Creates a new modular network with a specified number of nodes and parameters (`r`, `beta`, `c`). The network structure is generated using `ringity`, and its adjacency matrix and ring score are computed.

- `MyModInstance.run(n_runs=10, dt=0.001, T=1000)`:
  Runs a single Kuramoto simulation on the network, returning the result.




### Saving and Managing Results

- `MyModInstance.save_info(folder, verbose=True)`:
  Saves the network's parameters, adjacency matrix, and positions to the given folder.

- `Run.save_run(folder, verbose=False)`:
  Saves the results of a simulation run, including phase coherence, natural frequencies, and initial conditions.
