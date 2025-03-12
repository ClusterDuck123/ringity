import os
import json
import pandas as pd


class NetworkLoader:
    def __init__(self, folder, verbose=False):
        self.folder = folder
        self.verbose = verbose
        self.n_nodes = None
        self.r = None
        self.beta = None
        self.density = None
        self.ring_score = None
        self.adj_mat = None
        self.runs = []
        self.load_info()

    def load_info(self):
        """Load network parameters and run data from the given folder."""

        # Load network parameters
        with open(f"{self.folder}/network_params.json", "r") as fp:
            params = json.load(fp)
            self.n_nodes = params["n_nodes"]
            self.r = params["r"]
            self.beta = params["beta"]
            self.c = params["c"]
            self.ring_score = params["ring_score"]

        # Load adjacency matrix if verbose
        if self.verbose:
            self.adj_mat = pd.read_csv(f"{self.folder}/network.csv", index_col=0).values
            self.positions = pd.read_csv(
                f"{self.folder}/positions.csv", index_col=0
            ).values

        # Load run data
        runs_folder = f"{self.folder}/runs"
        for run_id in os.listdir(runs_folder):
            run_path = os.path.join(runs_folder, run_id)
            if os.path.isdir(run_path):
                run_data = {}

                # Load run parameters
                with open(f"{run_path}/run_params.json", "r") as fp:
                    run_params = json.load(fp)
                    run_data["dt"] = run_params["dt"]
                    run_data["T"] = run_params["T"]
<<<<<<< HEAD
                
                run_data["natfreqs"] = pd.read_csv(f"{run_path}/natfreqs.csv").values
                
=======
                    run_data["natfreqs"] = run_params["natfreqs"]

>>>>>>> origin/kuramoto
                # Load run summary if available
                summary_path = f"{run_path}/run_summary.json"
                if os.path.exists(summary_path):
                    with open(summary_path, "r") as fp:
                        summary_params = json.load(fp)
                        run_data["mean"] = summary_params["mean"]
                        run_data["std"] = summary_params["std"]
                        run_data["terminal_length"] = summary_params["terminal_length"]

                # Load full activity data if verbose
                if self.verbose:
                    activity_path = f"{run_path}/full.csv"
                    if os.path.exists(activity_path):
                        run_data["activity"] = pd.read_csv(
                            activity_path, index_col=0
                        ).values

                self.runs.append(run_data)
