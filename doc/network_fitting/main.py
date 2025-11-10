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
from ringity.networkfitting.retrieve_positions import PositionGraph
from ringity.networkfitting.fitting_model import FitNetwork

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


def main(network_name, network_model, make_figures, folder, unique):
    if unique:
        suffix = str(uuid.uuid4())
    else:
        suffix = "test"
    dirname = Path(folder) / f"{network_name}_{network_model}_{suffix}"

    os.makedirs(dirname, exist_ok=True)

    G_true = load_network(network_name)
    G, parameters = make_similar_network_model_random(G_true, network_model)
    positions, fitter = run_analysis(G)
    parameters["choice"] = {
        "network_name": network_name,
        "network_model": network_model,
        "suffix": suffix,
    }
    filename = f"{suffix}.json"
    save_values_to_json(
        folder=dirname,
        filename=filename,
        parameters=parameters,
        fitter_c=fitter.c,
        fitter_w=fitter.w,
    )

    if make_figures:
        if network_model == "none":
            color = COLOR_DICT[network_name]
        else:
            color = "k"
        save_figures(dirname, G, fitter, positions, color=color)


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


def run_analysis(G):
    self = PositionGraph(G)
    # self.positions = positions
    k = 0.1 * np.sqrt(2 * np.pi / len(self.nodelist))
    self.make_circular_spring_embedding(k=k, verbose=True)
    self.smooth_neighborhood_widths()
    self.recenter_and_reorient()
    self.reconstruct_positions()

    fitter = FitNetwork(self.G, self.rpositions)
    fitter.links_by_distance()

    def slope_down(theta, c, w):
        temp = w - theta
        return c * 0.5 * (np.abs(temp) + temp)

    p, _ = scipy.optimize.curve_fit(
        slope_down,
        fitter.midpoints[1:],
        50 * (fitter.counts_neighbors / fitter.counts_total)[1:],
        p0=None,
        sigma=None,
    )

    fitter.c, fitter.w = p
    fitter.c = fitter.c / 50

    return self, fitter


def save_values_to_json(folder, filename, parameters, fitter_c, fitter_w):
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

    data = {
        "parameters": parameters,
        "fitter_c": fitter_c,
        "fitter_w": fitter_w,
    }

    os.makedirs(folder, exist_ok=True)
    file_path = folder / filename

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"saved to {file_path}")


def edge_points(ax, G, p_dict, fontsize=15, color="k"):

    xy = np.array([[p_dict[i], p_dict[j]] for i, j in G.edges()])
    swap = np.vstack([xy[:, 1], xy[:, 0]]).T
    xy = np.vstack([xy, swap])

    ax.scatter(xy[:, 0], xy[:, 1], c=color, s=50 / np.sqrt(len(G.edges())))

    ax.set_xlabel(r"position of node $i$", fontsize=fontsize)
    ax.set_ylabel(r"position of node $j$", fontsize=fontsize)

    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(
        ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
        fontsize=fontsize,
    )
    ax.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_yticklabels(
        ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
        fontsize=fontsize,
    )


def save_figures(folder, G, fitter, self, fontsize=20, color="k"):
    fig, ax = plt.subplots()
    edge_points(
        ax=ax,
        G=self.G,
        p_dict=self.embedding_dict,
        fontsize=fontsize,
        color=color,
    )
    ax.axis("tight")
    fig.savefig(
        f"{folder}/raw_embedding.png", dpi=500, bbox_inches="tight", transparent=True
    )

    fig, ax = plt.subplots()
    edge_points(
        ax=ax,
        G=self.G,
        p_dict=self.rpositions,
        fontsize=fontsize,
        color=color,
    )
    ax.axis("tight")
    fig.savefig(
        f"{folder}/recovered_positions.png",
        dpi=500,
        bbox_inches="tight",
        transparent=True,
    )

    fig1 = fitter.links_by_distance()

    def slope_down(theta, c, w):
        temp = w - theta
        return c * 0.5 * (np.abs(temp) + temp)

    # [1:] because we want to avoid counting the absence of position_finder-loops
    p, _ = scipy.optimize.curve_fit(
        slope_down,
        fitter.midpoints[1:],
        50 * (fitter.counts_neighbors / fitter.counts_total)[1:],
        p0=None,
        sigma=None,
    )

    fitter.c, fitter.w = p
    fitter.c = fitter.c / 50

    fig2 = fitter.neighbor_proportion()
    fig3 = fitter.draw_edge_edge_and_fit()

    fig1.savefig(
        f"{folder}/links_by_distance.png",
        dpi=500,
        bbox_inches="tight",
        transparent=True,
    )
    fig2.savefig(
        f"{folder}/neighbor_proportion.png",
        dpi=500,
        bbox_inches="tight",
        transparent=True,
    )
    fig3.savefig(
        f"{folder}/draw_edge_edge_and_fit.png",
        dpi=500,
        bbox_inches="tight",
        transparent=True,
    )

    fig, ax = plt.subplots()
    ax.scatter(
        (fitter.counts_neighbors / fitter.counts_total)[1:],
        slope_down(fitter.midpoints[1:], fitter.c, fitter.w),
        c=color,
        s=300,
    )
    ax.set_xlabel("True Proportion")
    ax.set_ylabel("Fit Function")
    ax.axis("tight")
    fig.savefig(
        f"{folder}/goodness_of_fit.png", dpi=500, bbox_inches="tight", transparent=True
    )

    fig, ax = plt.subplots()
    ax.hist(
        self.rpositions.values(),
        density=True,
        color=color,
        bins=int(np.sqrt(len(self.rpositions.values()))),
    )
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(
        ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
        fontsize=fontsize,
    )

    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_yticklabels([0, 0.1, 0.2, 0.3], fontsize=fontsize)

    ax.set_xlabel("Position", fontsize=fontsize)
    ax.set_ylabel("Density", fontsize=fontsize)
    ax.axis("tight")
    fig.savefig(
        f"{folder}/rpos_bins.png", dpi=500, bbox_inches="tight", transparent=True
    )

    fig, ax = plt.subplots()
    ax.scatter(
        (fitter.counts_neighbors / fitter.counts_total)[1:],
        slope_down(fitter.midpoints[1:], fitter.c, fitter.w),
        c=color,
        s=300,
    )
    ax.set_xlabel("True Proportion")
    ax.set_ylabel("Fit Function")
    ax.axis("tight")
    fig.savefig(
        f"{folder}/goodness_of_fit.png", dpi=500, bbox_inches="tight", transparent=True
    )

    fig, ax = plt.subplots()
    ax.hist(
        self.rpositions.values(),
        density=True,
        color=color,
        bins=int(np.sqrt(len(self.rpositions.values()))),
    )
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(
        ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
        fontsize=fontsize,
    )

    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_yticklabels([0, 0.1, 0.2, 0.3], fontsize=fontsize)

    ax.set_xlabel("Position", fontsize=fontsize)
    ax.set_ylabel("Density", fontsize=fontsize)

    ax.axis("tight")

    fig.savefig(
        f"{folder}/fig_3_fit.png", dpi=500, bbox_inches="tight", transparent=True
    )
    print(f"saved to {folder}")


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
    parser.add_argument("--output_folder", default="test")
    parser.add_argument("--unique", default="true", type=is_true)

    args = parser.parse_args(sys.argv[1:])

    print(args)

    main(
        network_name=args.network,
        network_model=args.model,
        make_figures=args.make_figs,
        folder=args.output_folder,
        unique=args.unique,
    )
