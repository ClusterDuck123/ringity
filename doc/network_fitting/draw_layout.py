#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import json
import scipy.stats
import numpy as np
import ringity as rg
import networkx as nx
import matplotlib.pyplot as plt
import uuid

import sys
import os

from pathlib import Path

import json
import os
import sys

# ---------------------- GLOBAL VARIABLES ----------------------

EMPIRICALNET_DIR = Path("data/empirical_networks")
COLOR_SCHEME = {
    "Jasmine": "#ffd07b",
    "Glaucous": "#577399",
    "Dark purple": "#412234",
    "Moss green": "#748e54",
    "Keppel": "#44bba4",
}
COLOR_DICT = {
    "immune": COLOR_SCHEME["Keppel"],
    "fibro": COLOR_SCHEME["Glaucous"],
    "gene": COLOR_SCHEME["Moss green"],
    "lipid": COLOR_SCHEME["Jasmine"],
    "soil": COLOR_SCHEME["Dark purple"],
    "gene_corrected": COLOR_SCHEME["Moss green"],
}

# ------------------------ FUNCTIONS --------------------------


def main(network_name, network_model, folder, unique):
    if unique:
        suffix = str(uuid.uuid4())
    else:
        suffix = "test"
    dirname = Path(folder) / f"{network_model}_{suffix}"

    os.makedirs(dirname, exist_ok=True)

    color = COLOR_DICT[network_name]

    G_true = load_network(network_name)
    G, parameters = make_similar_network_model_random(G_true, network_model)

    pos = nx.spectral_layout(G)
    pos = nx.spring_layout(G, pos=pos)

    n = len(G.nodes())
    node_size = 1000 / np.sqrt(n)
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_color=color,
        edgecolors="k",
        linewidths=1.0,
        node_size=node_size,
    )
    nx.draw_networkx_edges(G, pos=pos, ax=ax, alpha=0.1)

    ax.axis("off")

    print(f"{dirname}/spring_layout.png")
    fig.savefig(
        f"{dirname}/spring_layout.png", dpi=300, bbox_inches="tight", transparent=True
    )


def load_network(name):
    G = nx.read_gml(EMPIRICALNET_DIR / f"{name}.gml")
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.relabel_nodes(G, str)
    return G


def get_largest_component_with_positions(G, positions):
    largest_cc = max(nx.connected_components(G), key=len)
    G_largest = G.subgraph(largest_cc).copy()
    largest_cc_nodes = list(largest_cc)
    largest_positions = [positions[node] for node in largest_cc_nodes]
    return G_largest, largest_positions


def make_similar_network_model_random(G_true, network_model):
    avg_degree = np.mean(list(dict(G_true.degree()).values()))

    N = nx.number_of_nodes(G_true)
    density = nx.density(G_true)

    if network_model == "none":
        G = G_true
        parameters = {}

    if network_model == "erdos_renyi":
        p = density
        parameters = {"p": p}
        G = nx.erdos_renyi_graph(N, p)

    if network_model == "configuration":
        G = nx.configuration_model(list(dict(G_true.degree()).values()))
        parameters = {}

    if network_model == "this_paper":
        beta = 0.7 + 0.3 * np.random.rand()
        r = 0.5 * np.random.rand()

        G, positions = rg.network_model(
            N, rho=density, beta=beta, r=r, return_positions=True
        )
        G, positions = get_largest_component_with_positions(G, positions)

        parameters = {"beta": beta, "r": r, "density": density}

    parameters["true_density"] = density
    parameters["true_avg_degree"] = avg_degree
    parameters["N"] = N

    return G, parameters


def is_true(x):
    return x.lower() == "true"


# --------------------------- MAIN -----------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="ProgramName",
        description="What the program does",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("--network", default="lipid")
    parser.add_argument("--model", default="none")
    parser.add_argument("--make_figs", default="true", type=is_true)
    parser.add_argument("--output_folder", default="figures/network_fitting/lipid/")
    parser.add_argument("--unique", default="true", type=is_true)

    args = parser.parse_args(sys.argv[1:])

    print(args)

    main(
        network_name=args.network,
        network_model=args.model,
        folder=args.output_folder,
        unique=args.unique,
    )
