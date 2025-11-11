import networkx as nx
import csv
import argparse
import ringity as rg
import numpy as np
import pandas as pd


def generate_network(network_type, n=30, k=4, p=0.1, beta=0.9, r=0.3, c=1.0):

    if network_type == "watts_strogatz":
        G = nx.watts_strogatz_graph(n=n, k=k, p=p)
        positions = 2 * np.pi * np.array(list(G.nodes())) / G.order()

    elif network_type == "newman_watts_strogatz":
        G = nx.newman_watts_strogatz_graph(n=n, k=k, p=p)
        positions = 2 * np.pi * np.array(list(G.nodes())) / G.order()

    elif network_type == "model_network":
        G, positions = rg.network_model(N=n, beta=beta, r=r, c=c, return_positions=True)
    else:
        raise ValueError(f"Unsupported network type: {network_type}")
    return G, positions


def save_graph_to_gml(G, filename):
    nx.write_gml(G, filename)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 1D ring-like network and export data."
    )
    parser.add_argument(
        "--type",
        type=str,
        default="model_network",
        choices=["watts_strogatz", "newman_watts_strogatz", "model_network"],
        help="Type of network to generate.",
    )
    parser.add_argument("--nodes", type=int, default=100, help="Number of nodes.")
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Each node is joined with its k nearest neighbors.",
    )
    parser.add_argument("--p", type=float, default=0.1, help="Rewiring probability.")
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9,
        help="Each node is joined with its k nearest neighbors.",
    )
    parser.add_argument("--c", type=float, default=1.0, help="Rewiring probability.")
    parser.add_argument("--r", type=float, default=0.3, help="Rewiring probability.")
    parser.add_argument(
        "--gml", type=str, default="test.gml", help="Path to output GML file."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="test.csv",
        help="Path to output CSV file for node positions.",
    )

    args = parser.parse_args()

    G, positions = generate_network(args.type, n=args.nodes, k=args.k, p=args.p)
    nodelist = sorted(list(G.nodes()))

    save_graph_to_gml(G, args.gml)
    pd.Series(positions, index=nodelist).to_csv(args.csv)


if __name__ == "__main__":
    main()
