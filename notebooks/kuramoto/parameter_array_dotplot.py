import matplotlib.pyplot as plt
import numpy as np
import ringity as rng
from itertools import product
from main import MyModInstance,Run
import argparse
import uuid
import os
import os
import pandas as pd
import tqdm


# ------------------------------------------------------------------------
# Global settings / theme
# ------------------------------------------------------------------------
rng.set_theme()

# Constants for dot_plot
DOT_SIZE = 1500
SCATTER_VMIN = 0.6
SCATTER_VMAX = 1.0

def main():
    
    parser = argparse.ArgumentParser(description="Create dotplot showing dependnce of synchronicity and coherence on parameter values of network construction")
    parser.add_argument("--i", type=str, default="test_network", help="The input folder containing the networks.")
    parser.add_argument("--o", type=str, default="test_network", help="output filename.")
    parser.add_argument("--terminal_length", type=int, default=10, help="terminal network.")
    parser.add_argument("--threshold", type=float, default=0.001, help="threshold for determining asynchronicity")

    args = parser.parse_args()
    
    
    terminal_length = args.terminal_length
    threshold  = args.threshold 
    
    input_folder = args.i
    
    fig = load_data_and_create_figure(input_folder,terminal_length, threshold)
    
    output_file = args.o
    fig.savefig(output_file)
    
def load_data_and_create_figure(top_folder,terminal_length, threshold):
    
    beta_centers,r_centers = np.linspace(0.8,1.0,5),np.linspace(0.1,0.25,4)
    n_beta_vals,n_r_vals = len(beta_centers),len(r_centers)
    
    parameter_network_dict = load_and_sort_networks_by_parameter(top_folder,beta_centers,r_centers)
    
    coherences = np.zeros((n_beta_vals,n_r_vals))
    fractions  = np.zeros((n_beta_vals,n_r_vals))
    
    for parameter_index_pair, networks in parameter_network_dict.items():
        
        
        
        i,j = parameter_index_pair
        runs = runs_from_networks(networks)
        
        
        
    
        fractions[i,j] = 1-proportion_asynch(runs,terminal_length, threshold)
        coherences[i,j] = average_terminal_phase_coherence(runs, terminal_length, threshold )
    
    fig = full_figure(coherences, fractions)
    return fig
        
        
        
    
    

def load_and_sort_networks_by_parameter(top_folder, beta_centers,r_centers):
    # load all the summaries networks and their runs
    # load them into a dict keyed by the network's parameter values
    
    beta_bin_starts = (beta_centers[:-1]+beta_centers[1:])/2
    r_bin_starts    = (r_centers[:-1]+r_centers[1:])/2
    

    
    print(beta_bin_starts)
    out = {}
    for subfolder in os.listdir(top_folder):
        folder = os.path.join(top_folder,subfolder)
        try:
            network = MyModInstance.load_instance(folder)
            network.folder = folder
            
        
            i = np.digitize(network.beta, beta_bin_starts)
            j = np.digitize(network.r, r_bin_starts)
            
            print(network.beta, beta_bin_starts)
            print(i)
            
            try:
                out[i,j].append(network)
            except KeyError:
                out[i,j] = [network]
            
        except FileNotFoundError as e:
            print(e)
            
    return out

def load_runs_from_folder(folder):
    
    out = []
    for subfolder in os.listdir(folder):
        full_subfolder_path = os.path.join(folder, subfolder)
        try:
            out.append(Run.load_run(full_subfolder_path))
        except Exception as e:
            print(e)
    return out
        

def is_asynch(run, terminal_length, threshold):
    return np.std(run.phase_coherence[-terminal_length:]) > threshold

def proportion_asynch(runs, terminal_length, threshold):
    classification = [is_asynch(run, terminal_length, threshold) for run in runs]
    return sum(classification)/len(classification)

def average_terminal_phase_coherence(runs, terminal_length, threshold):
    
    all_terminal_phase_coherence = []
    for run in runs:
        if not is_asynch(run,terminal_length, threshold):
            all_terminal_phase_coherence.append(np.mean(run.phase_coherence[-terminal_length:]))
    
    return np.mean(all_terminal_phase_coherence)
            
def runs_from_networks(networks):
    
    out = []
    for network in networks:
        run_folder = os.path.join(network.folder, "runs/")
        out.extend(load_runs_from_folder(run_folder))
    
    return out


        
    
def dot_plot(
    coherences: np.ndarray,
    fractions: np.ndarray,
    scale_by: str = "area",
    ax: plt.Axes = None,
    with_fraction_labels: bool = True,
    with_coherence_labels: bool = True,
) -> plt.Axes:
    """
    Draw a 'dot plot' where each point has two attributes:
      - A color corresponding to the coherence value.
      - A size (relative area or length) corresponding to the fraction value.

    Parameters
    ----------
    coherences : np.ndarray
        2D array of coherence values (must match shape of fractions).
    fractions : np.ndarray
        2D array of fraction values (must match shape of coherences).
    scale_by : {'area', 'length'}, optional
        How to scale the marker size by the fraction value:
          - 'area': scale marker's area by fraction (default)
          - 'length': scale marker's diameter by fraction
    ax : plt.Axes, optional
        Matplotlib axes object on which to draw the plot. If not provided,
        a new figure and axes are created.
    with_fraction_labels : bool, optional
        If True, displays fraction labels below each marker.
    with_coherence_labels : bool, optional
        If True, displays coherence labels centered on each marker.

    Returns
    -------
    plt.Axes
        The axes on which the plot is drawn.
    """
    assert coherences.shape == fractions.shape, (
        "coherences and fractions must have the same shape."
    )

    # Validate 'scale_by' input
    if scale_by not in {"area", "length"}:
        raise ValueError("scale_by must be one of {'area', 'length'}.")

    # Set up axes
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 6))
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, zorder=0)

    n_rows, n_cols = coherences.shape

    for row_idx, col_idx in product(range(n_rows), range(n_cols)):
        # Reverse indexing from the original code
        curr_cohe = coherences[-row_idx - 1, col_idx]
        curr_frac = fractions[-row_idx - 1, col_idx]

        # Determine how to scale the circle
        if scale_by == "area":
            # fraction scales area directly
            size_multiplier = curr_frac
        else:  # scale_by == "length"
            # fraction scales radius
            size_multiplier = curr_frac ** 2

        # Draw the outer white circle
        ax.scatter(
            col_idx, row_idx,
            s=DOT_SIZE,
            c="white",
            cmap="none",
            linewidth=1,
            edgecolors="black",
            zorder=2,
        )

        # Draw the scaled circle based on fraction
        ax.scatter(
            col_idx, row_idx,
            s=DOT_SIZE * size_multiplier,
            c=curr_cohe,
            cmap="cividis",
            linewidth=1,
            edgecolors="black",
            zorder=3,
            vmin=SCATTER_VMIN,
            vmax=SCATTER_VMAX,
        )

        # Optionally add text labels
        if with_fraction_labels:
            ax.text(
                col_idx,
                row_idx - 0.35,
                f"{round(100 * curr_frac)}%",
                ha="center",
                va="center",
                color="black",
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
            )

        if with_coherence_labels:
            ax.annotate(
                f"{curr_cohe:.2f}",
                (col_idx, row_idx),
                ha="center",
                va="center",
                color="black",
            )

    return ax


    
def full_figure(coherences, fractions):
    # --------------------------------------------------------------------
    # Create figure and draw the plot
    # --------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 6))
    dot_plot(
        coherences,
        fractions,
        scale_by="area",  # or "length"
        ax=ax,
        with_fraction_labels=True,
        with_coherence_labels=True,
    )

    # Add colorbar for coherence
    # We pass dummy scatter args because we only need the colorbar
    dummy_scatter = ax.scatter([], [], c=[], cmap="cividis", vmin=0.5, vmax=1.0)
    cbar = fig.colorbar(dummy_scatter, ax=ax)
    cbar.set_label(r"Avg. Coherence ($\langle R_{\infty} \rangle$)")

    # Axis settings
    n_rows, n_cols = coherences.shape
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.75, n_rows - 0.5)
    ax.set_xticks(
        range(n_cols),
        map("{:.2f}".format, np.linspace(0.10, 0.25, n_cols))
    )
    ax.set_yticks(
        range(n_rows),
        map("{:.2f}".format, np.linspace(0.80, 1.00, n_rows))
    )
    ax.set_xlabel(r"Response Length ($r$)")
    ax.set_ylabel(r"Delay Parameter ($\beta$)")
    ax.set_title("Synchronization Behaviour", size=24)

    return fig


def example_figure():
    # --------------------------------------------------------------------
    # Example data
    # --------------------------------------------------------------------
    coherences = np.array([
        [0.00, 0.05, 0.11, 0.16],
        [0.21, 0.26, 0.32, 0.37],
        [0.42, 0.47, 0.53, 0.58],
        [0.63, 0.68, 0.74, 0.79],
        [0.84, 0.89, 0.95, 1.00],
    ])

    fractions = np.array([
        [0.00, 0.05, 0.11, 0.16],
        [0.21, 0.26, 0.32, 0.37],
        [0.42, 0.47, 0.53, 0.58],
        [0.63, 0.68, 0.74, 0.79],
        [0.84, 0.89, 0.95, 1.00],
    ])
    
    fig = full_figure(coherences, fractions)
    plt.show()
    

if __name__ == "__main__":
    main()
