from main import MyModInstance, Run
import argparse
import numpy as np
import pandas as pd
import tqdm
import os


def decide_run_type(run, coherence_threshold, asynchrony_threshold):

    if run.terminal_std > asynchrony_threshold:
        return "asynch"

    elif run.terminal_mean < 1 - coherence_threshold:
        return "traveling"

    else:
        return "coherent"


def rerun_and_save(network, run, output_folder, run_type):

    print("running new instance...")
    new_run = network.run(dt=args.dt, T=args.T)
    new_run.natfreqs = run.natfreqs
    new_run.initialize_simulation()

    new_network_folder = os.path.join(output_folder, f"{run_type}_{network.folder}")
    network.save_info(new_network_folder)
    new_run.save_run(new_network_folder, verbose=True)


def main(args):
    """Generate and save instances of the network model with varying beta and r parameters."""

    target_number = args.target_number
    input_folder = args.i
    output_folder = args.o

    # terminal_length      = args.terminal_length
    # coherence_threshold  = args.coherence_threshold
    # asynchrony_threshold = args.asynchrony_threshold

    # print()

    # n_networks_saved = {"asynch":0, "traveling":0, "coherent":0}

    print("scanning ", input_folder)

    out = []
    for network_folder in os.listdir(
        input_folder
    ):  # tqdm.tqdm(os.listdir(input_folder)):

        print("     in ", network_folder)

        folder = os.path.join(input_folder, network_folder)
        network = MyModInstance.load_instance(folder)
        network.folder = network_folder
        network.fullpath = folder

        run_folder = os.listdir(f"{folder}/runs/")[0]
        print("          in ", run_folder)
        run_path = os.path.join(folder, "runs", run_folder)

        run = Run.load_run(run_path)

        print("extracted run!")
        out.append([folder, run_folder, run.terminal_std, run.terminal_mean])
        print([folder, run_folder, run.terminal_std, run.terminal_mean])

    summary_stats = pd.DataFrame(
        out, columns=["network_folder", "folder", "terminal_std", "terminal_mean"]
    )
    summary_stats.to_csv("test.csv")

    print(summary_stats)

    for i, v in (
        summary_stats.sort_values("terminal_mean").iloc[:target_number].iterrows()
    ):

        run_type = "travelling"

        network_folder = v["network_folder"]
        run_folder = v["folder"]

        network = MyModInstance.load_instance(network_folder)

        run_path = os.path.join(folder, "runs", run_folder)
        print(run_path)
        assert os._exists(run_path)
        run = Run.load_run(run_path)

        rerun_and_save(network, run, output_folder, run_type)

    for i, v in (
        summary_stats.sort_values("terminal_mean").iloc[-target_number:].iterrows()
    ):

        run_type = "coherent"

        network_folder = v["network_folder"]
        run_folder = v["folder"]

        network = MyModInstance.load_instance(network_folder)

        run_path = os.path.join(folder, "runs", run_folder)
        run = Run.load_run(run_path)

        rerun_and_save(network, run, output_folder, run_type)

    for i, v in (
        summary_stats.sort_values("terminal_std").iloc[-target_number:].iterrows()
    ):

        run_type = "asynchronous"

        network_folder = v["network_folder"]
        run_folder = v["folder"]

        network = MyModInstance.load_instance(network_folder)

        run_path = os.path.join(folder, "runs", run_folder)
        run = Run.load_run(run_path)

        rerun_and_save(network, run, output_folder, run_type)

    """
    for network_folder in tqdm.tqdm(os.listdir(input_folder)):

        try:
            folder = os.path.join(input_folder,network_folder)
            network = MyModInstance.load_instance(folder)
            network.folder = network_folder
            network.fullpath = folder

            run_folder = os.listdir(f"{folder}/runs/")[0]
            run_path = os.path.join(folder, "runs", run_folder)

            run = Run.load_run(run_path)

            print(run.natfreqs)
            print(run.adj_mat)
            print(run.init_conditions)


            run_type = decide_run_type(run,
                                       coherence_threshold,
                                       asynchrony_threshold
                                       )




            if n_networks_saved[run_type] < target_number:

                print(run_type)
                print("running new instance...")
                new_run = network.run(dt=args.dt,T=args.T)
                #new_run.natfreqs
                #terminal_length= run.natfreqs
                new_run.initialize_simulation()


                new_network_folder = os.path.join(output_folder, f"{run_type}_{network.folder}")
                network.save_info(new_network_folder)
                new_run.save_run(new_network_folder,verbose=True)

                n_networks_saved[run_type] += 1

        except FileNotFoundError as e:

            print(e)
        except IndexError as e:

            print(e)

    print(n_networks_saved)
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Kuramoto simulation.")

    parser = argparse.ArgumentParser(description="Generate and save a network.")

    parser.add_argument(
        "--target_number",
        type=int,
        default=3,
        help="Number of runs of each type to select and retry.",
    )
    parser.add_argument(
        "--i", type=str, default="data/parameter_array/", help="The input folder"
    )
    parser.add_argument(
        "--o", type=str, default="data/rerun_pa/", help="The output folder"
    )

    # parser.add_argument("--terminal_length",      type=int,   default=100,   help="The number of final timesteps to take when classifying final behaviour.")
    # parser.add_argument("--coherence_threshold",  type=float, default=0.5, help="Runs with final mean phase_coherence above 1 - coherence_threshold are coherent.")
    # parser.add_argument("--asynchrony_threshold", type=float, default=0.000001, help="Runs with final st.dev. in phase_coherence above asynchrony_threshold are asynchronous.")
    # previously default=0.0000000001

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
        "--verbose", type=bool, default=False, help="how  much info is saved"
    )

    args = parser.parse_args()

    main(args)
