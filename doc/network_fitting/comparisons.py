#!/usr/bin/env python
# coding: utf-8

# In[895]:


import os
import tqdm
import json
import scipy.stats
import numpy as np
import pandas as pd
import ringity as rg
import networkx as nx
import itertools as it
import matplotlib.pyplot as plt
import uuid

from retrieve_positions import PositionGraph

import json
import os


def save_values_to_json(
    folder, filename, density, avg_degree, N, parameters, fitter_c, fitter_w
):
    """
    Saves the given values to a JSON file within the specified folder and filename structure.

    Args:
        folder (str): The name of the main folder.
        filename (str): The name of the filename.
        density (float): The density value.
        avg_degree (float): The average degree value.
        N (int): The N value.
        beta (float): The beta value.
        c (float): The c value.
        r (float): The r value.
        fitter_c (float): The fitter_c value.
        fitter_w (float): The fitter_w value.
    """

    # Create the folder structure if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Create the JSON file path
    file_path = os.path.join(folder, filename)

    # Create a dictionary to store the values
    data = {
        "density": density,
        "avg_degree": avg_degree,
        "N": N,
        "parameters": parameters,
        "fitter_c": fitter_c,
        "fitter_w": fitter_w,
    }

    # Write the dictionary to the JSON file
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def extract_and_save_info(folder, filename, G, parameters):

    N = len(G.nodes())
    density = 2 * G.size() / (N * (N - 1))
    avg_degree = np.mean(list(dict(G.degree()).values()))

    self = PositionGraph(G)
    k = np.sqrt(2 * np.pi / len(self.nodelist))
    self.make_circular_spring_embedding(k=k, verbose=True)

    self.smooth_neighborhood_widths()
    self.recenter_and_reorient()
    self.reconstruct_positions()

    from ringity.network_fitting.fitting_model import FitNetwork

    def slope_down(theta, c, w):
        temp = w - theta
        return c * 0.5 * (np.abs(temp) + temp)

    fitter = FitNetwork(self.G, self.rpositions)

    fig1 = fitter.links_by_distance()

    p, _ = scipy.optimize.curve_fit(
        slope_down,
        fitter.midpoints[1:],
        50 * (fitter.counts_neighbors / fitter.counts_total)[1:],
        p0=None,
        sigma=None,
    )

    fitter.c, fitter.w = p
    fitter.c = fitter.c / 50

    save_values_to_json(
        folder,
        filename,
        density,
        avg_degree,
        N,
        parametenotebooks / network_fitting / comparisons.pyrs,
        fitter.c,
        fitter.w,
    )


def remove_selfloops(G):
    for u, v in G.edges():
        if u == v:
            G.remove_edge(u, v)


G_lipid = nx.read_gml("lipid.gml")
G_lipid = G_lipid.subgraph(  # get lcc, in case threshold was chosen too high
    max(nx.connected_components(G_lipid), key=len)
).copy()
remove_selfloops(G_lipid)
G_lipid = nx.relabel_nodes(G_lipid, lambda x: str(x))

G_fibro = nx.read_gml("fibro.gml")
G_fibro = G_fibro.subgraph(
    max(nx.connected_components(G_fibro), key=len)
).copy()  # get lcc, in case threshold was chosen too low
remove_selfloops(G_fibro)
G_fibro = nx.relabel_nodes(G_fibro, lambda x: str(x))

# 2c Immune Network
G_immune = nx.read_gml("immune.gml")
remove_selfloops(G_immune)
G_immune = nx.relabel_nodes(G_immune, lambda x: str(x))

# 2d Circadian Network
# G_genes = nx.read_gml("gene.gml")
# remove_selfloops(G_genes)

# 2e Soil Network
G_soil = nx.read_gml("soil.gml")
remove_selfloops(G_soil)
G_soil = nx.relabel_nodes(G_soil, lambda x: str(x))

# Dictionary to hold all networks
networks = {
    "lipid": G_lipid,
    "fibro": G_fibro,
    "immune": G_immune,
    # "genes": G_genes,  # Uncomment if needed after processing
    "soil": G_soil,
}


def get_largest_component_with_positions(G, positions):
    # Find the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)

    # Subset the graph to the largest connected component
    G_largest = G.subgraph(largest_cc).copy()

    # Get the node indices for the largest component
    largest_cc_nodes = list(largest_cc)

    # Get the positions corresponding to the nodes in the largest component
    largest_positions = [positions[node] for node in largest_cc_nodes]

    return G_largest, largest_positions


import matplotlib

matplotlib.use("MacOSX")


for name, G_true in networks.items():

    avg_degree = np.mean(list(dict(G_true.degree()).values()))
    avg_degree

    N = len(G_true.nodes())
    density = 2 * G_true.size() / (N * (N - 1))

    folder = f"imitations_{name}/"

    os.makedirs(folder, exist_ok=True)
    os.makedirs(folder + "ER", exist_ok=True)
    os.makedirs(folder + "OM", exist_ok=True)
    os.makedirs(folder + "CM", exist_ok=True)

    extract_and_save_info(folder, f"true.json", G_true, {})

    for _ in range(100):

        try:
            uuid_ = str(uuid.uuid4())
            true_positions = False
            network_type = "positionless"

            p = density
            parameters = {"p": p}

            G = nx.erdos_renyi_graph(N, p)

            filename = f"ER/N_{N}_p_{round(p*100)}_{uuid_}.json"

            extract_and_save_info(folder, filename, G, parameters)
        except Exception as e:
            print(e)

    for _ in range(100):

        try:
            uuid_ = str(uuid.uuid4())
            true_positions = False
            network_type = "positionless"

            G = nx.configuration_model(list(dict(G_true.degree()).values()))

            filename = f"CM/iteration_{uuid_}.json"

            extract_and_save_info(folder, filename, G, parameters)

        except Exception as e:
            print(e)

    for _ in range(100):

        try:
            uuid_ = str(uuid.uuid4())
            beta = 0.7 + 0.3 * np.random.rand()
            r = 0.5 * np.random.rand()

            G, positions = rg.network_model(
                N, density=density, beta=beta, r=r, return_positions=True
            )

            network_type = "positional"
            filename = f"OM/network_model_N_{N}_density_{round(density*100)}_r_{round(r*100)}_beta_{round(beta*100)}_{uuid_}.json"
            os.makedirs(folder, exist_ok=True)

            G, positions = get_largest_component_with_positions(G, positions)

            parameters = {"beta": beta, "r": r, "density": density}

            extract_and_save_info(folder, filename, G, parameters)
        except Exception as e:
            print(e)
