import os
from load_saved_info import NetworkLoader
import pandas as pd

# load all the summaries networks and their runs
# load them into a dict keyed by the network's parameter values
out = {}
for subfolder in os.listdir("data/concise/parameter_array_outdated"):
    folder = f"data/concise/parameter_array_outdated/{subfolder}"
    try:
        loader = NetworkLoader(folder, verbose=False)
        try:
            out[round(loader.beta, 3), round(loader.r, 3)].append(loader)
        except KeyError:
            out[round(loader.beta, 3), round(loader.r, 3)] = [loader]
    except FileNotFoundError:
        print("ow!")
network_array_dict = out

out = []
for param_pair, network_list in network_array_dict.items():

    n_total = len(network_list)
    n_asynchronous = 0
    n_coherent = 0
    n_traveling = 0

    for network in network_list:
        std = network.runs[0]["std"]
        mean = network.runs[0]["mean"]

        if std > 0.00001:

            if mean > 0.999999:
                n_coherent += 1
            else:
                n_traveling += 1
        else:
            n_asynchronous += 1

        out.append(
            [
                param_pair[0],
                param_pair[1],
                n_total,
                n_asynchronous,
                n_coherent,
                n_traveling,
            ]
        )
pd.DataFrame(
    out, columns=["beta", "r", "n_total", "n_asynchronous", "n_coherent", "n_traveling"]
)
