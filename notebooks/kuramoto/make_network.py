import ringity.ringscore
from core import MyModInstance

import itertools as it
import argparse
import ringity
import tqdm
import uuid
import os


def make_and_save_network(output_folder, n_nodes, r, beta, c):

    while True:
        try:
            graph_obj = MyModInstance(n_nodes=n_nodes, r=r, beta=beta, c=c)
            folder = os.path.join(output_folder, f"network_{uuid.uuid4()}")
            os.makedirs(folder, exist_ok=True)
            graph_obj.save_info(folder, verbose=True)
            print("network written to... ", folder)
            break
        except ringity.utils.exceptions.DisconnectedGraphError:
            print("Network disconnected, retrying...")


def make_parameter_array(output_folder):
    beta_values = [0.8, 0.85, 0.9, 0.95, 1.0]
    r_values = [0.1, 0.15, 0.2, 0.25]

    param_pairs = list(it.product(r_values, beta_values))

    for _, (r, beta) in tqdm.tqdm(
        it.product(range(50), param_pairs), total=50 * len(param_pairs)
    ):

        make_and_save_network(output_folder, 200, r, beta, 1.0)


def main():
    """Generate and save instances of the network model with varying beta and r parameters.
    Example usage: python make_network.py --output_folder test/ --n_nodes 50 --r 0.3 --beta 0.9 --c 1.0
    """
    parser = argparse.ArgumentParser(description="Generate and save a network.")
    parser.add_argument("--n_nodes", type=int, required=True, help="Number of nodes.")
    parser.add_argument(
        "--r",
        type=float,
        required=True,
        help="The r parameter of the network model, that controls homophily",
    )
    parser.add_argument(
        "--beta",
        type=float,
        required=True,
        help="The beta parameter of the network model, that controls distribution of delays",
    )
    parser.add_argument(
        "--c",
        type=float,
        required=True,
        help="The  c parameter of the network model, that controls overall interaction probability of nodes",
    )
    parser.add_argument(
        "--output_folder", type=str, required=True, help="The output folder"
    )

    args = parser.parse_args()

    make_and_save_network(args.output_folder, args.n_nodes, args.r, args.beta, args.c)


main()
