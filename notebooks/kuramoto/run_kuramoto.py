import ringity.ringscore
from core import MyModInstance

import itertools as it
import argparse
import ringity
import tqdm
import uuid
import pandas as pd
from pathlib import Path
import os
import fcntl


def append_to_summary_file(network, run, summary_info_file, terminal_length=200):

    run.calculate_stats(terminal_length=terminal_length)
    try:
        network.folder
    except AttributeError:
        network.folder = "unsaved"

    try:
        run.run_folder
    except AttributeError:
        run.run_folder = "unsaved"

    info = {
        "n_nodes": network.n_nodes,
        "r": network.r,
        "beta": network.beta,
        "c": network.c,
        "ring_score": network.ring_score,
        "network_folder": network.folder,
        "terminal_length": terminal_length,
        "terminal_std": run.terminal_std,
        "terminal_mean": run.terminal_mean,
        "run_folder": run.run_folder,
        "dt": run.dt,
        "T": run.T,
    }

    print(info)
    info = pd.DataFrame([info])

    output_csv = Path(summary_info_file)

    if output_csv.is_file():

        fp = open(output_csv, "a")

        fcntl.flock(fp, fcntl.LOCK_EX)
        info.to_csv(path_or_buf=fp, mode="a", header=None)
        fcntl.flock(fp, fcntl.LOCK_UN)
    else:
        fp = open(output_csv, "w")

        fcntl.flock(fp, fcntl.LOCK_EX)
        info.to_csv(path_or_buf=fp)
        fcntl.flock(fp, fcntl.LOCK_UN)

    print("summary appended to", summary_info_file)


def main():
    """Main function to run batch simulations of Kuramoto models on loaded networks.
    python load_network_run_kuramoto.py --i test5/network_673a0191-e9f5-45b7-8c26-127a5d278b6f/ --sumarry-info-file foo.csv
    """
    parser = argparse.ArgumentParser(description="Run Kuramoto simulation.")
    parser.add_argument(
        "--i", type=str, default="test_network", help="The input and output folder"
    )
    parser.add_argument(
        "--summary-info-file",
        default="none",
        help="append compressed summary to this file",
    )
    parser.add_argument(
        "--T",
        type=float,
        default=1000,
        help="Time to run the system for (not the number of timesteps! that is floor(T/dt))",
    )
    parser.add_argument(
        "--dt", type=float, default=0.001, help="time interval per step"
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="save more information, including full activity matrix of run",
    )
    parser.add_argument(
        "--quiet",
        type=bool,
        default=False,
        help="save minimal information, not enough to re-run the calculation",
    )
    parser.add_argument(
        "--only-summary",
        type=bool,
        default=False,
        help="Only save to the summary info file, not to a folder",
    )
    parser.add_argument(
        "--terminal_length",
        type=int,
        default=200,
        help="numer of timpoeints from end to use to calculate terminal mean and terminal std",
    )

    args = parser.parse_args()

    input_folder = args.i

    network = MyModInstance.load_instance(input_folder)

    run = network.run(T=args.T, dt=args.dt)

    if not args.only_summary:
        run.save_run(
            input_folder,
            verbose=args.verbose,
            quiet=args.quiet,
            terminal_length=args.terminal_length,
        )
        print(
            "saved info to ",
            input_folder,
            "/runs/",
            run.run_id,
            "quiet ",
            args.quiet,
            "verbose",
            args.verbose,
        )

    if args.summary_info_file != "none":
        append_to_summary_file(
            network, run, args.summary_info_file, args.terminal_length
        )


main()
