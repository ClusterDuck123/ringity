"""
Position Graph Analysis Module

This module provides tools for analyzing network topology through circular embeddings.
It implements algorithms to embed graph nodes on a circle and analyze their positional
relationships to understand network structure and connectivity patterns.

Key Concepts:
- Circular Spring Embedding: Places nodes on a circle using force-directed layout
- Position Reconstruction: Adjusts embeddings based on local neighborhood density

The main workflow:
1. Create circular spring embedding of network nodes
3. Recenter and reorient the embedding to find optimal positioning
2. Analyze neighborhood widths around the circle
4. Reconstruct final positions using inverse CDF transformation

Classes:
    PositionGraph: Main class for circular network embedding and analysis

Functions:
    circular_mean: Compute circular (angular) mean of angles
    estimate_std: Estimate standard deviation with sliding window
    reconstruct_icdf: Reconstruct inverse cumulative distribution function
"""

import pandas as pd
import numpy as np
import tqdm
import networkx as nx
from scipy.ndimage import convolve
from pathlib import Path
import os


def circular_mean(theta):
    """
    Compute the circular mean of a set of angles.

    The circular mean is the mean direction of a set of angles, accounting for
    the circular (periodic) nature of angular data where 0 and 2π are equivalent.

    Args:
        theta (array-like): Array of angles in radians

    Returns:
        float: Circular mean angle in radians, normalized to [0, 2π)

    Example:
        >>> angles = np.array([0.1, 6.2, 0.2])  # angles near 0
        >>> circular_mean(angles)  # Returns ~0.17, not ~2.17
    """
    x = np.mean(np.cos(theta))
    y = np.mean(np.sin(theta))
    return np.mod(np.arctan2(y, x), 2 * np.pi)


def estimate_std(array_of_values, window_size=30):
    """
    Estimate standard deviation using a sliding window approach over circular data.

    This function computes standard deviation estimates for a sequence of sample arrays
    using a sliding window, smoothing noisy variance
    estimates in circular embeddings where adjacent positions should have similar variances.

    Args:
        array_of_values (list of numpy arrays): List where each element contains
                                                sample values for std estimation
        window_size (int, optional): Size of sliding window. Default is 30.

    Returns:
        numpy.ndarray: Array of estimated standard deviations for each position

    Notes:
        - Uses wrap-around convolution treating the data as circular
        - Applies Bessel's correction (N-1 denominator) for unbiased variance
        - Useful for smoothing neighborhood width estimates in circular embeddings

    Example:
        >>> # Simulate neighborhood samples for each position around a circle
        >>> data = [np.random.randn(np.random.randint(5, 15)) for _ in range(100)]
        >>> smoothed_stds = estimate_std(data, window_size=20)
    """
    count = []
    sum_value = []
    sum_square = []

    for samples in array_of_values:
        count.append(len(samples))
        sum_value.append(sum(samples))
        sum_square.append(sum(samples**2))

    # Apply sliding window smoothing with circular boundary conditions
    smooth_count = convolve(count, np.ones(window_size), mode="wrap")
    smooth_sum_value = convolve(sum_value, np.ones(window_size), mode="wrap")
    smooth_sum_square = convolve(sum_square, np.ones(window_size), mode="wrap")

    # Bessel's correction for unbiased variance estimate
    correction = (smooth_count - 1) / smooth_count

    return np.sqrt(
        correction
        * (smooth_sum_square / smooth_count - (smooth_sum_value / smooth_count) ** 2)
    )


def reconstruct_icdf(u_samples, ipdf_values):
    """
    Reconstruct inverse cumulative distribution function from samples and density.

    This function reconstructs the ICDF by integrating the interpolated PDF over
    intervals defined by the sample points. Used to transform uniform circular
    positions to positions that account for varying local density.

    Args:
        u_samples (numpy.ndarray): Sample points from distribution (typically [0,1])
        ipdf_values (numpy.ndarray): Interpolated PDF values at sample points

    Returns:
        tuple: (u_midpoints, cumulative_icdf)
            - u_midpoints: Interval boundaries for piecewise linear interpolation
            - cumulative_icdf: Cumulative ICDF values for position transformation

    Example:
        >>> # Uniform samples with varying density
        >>> u = np.linspace(0, 1, 50)
        >>> pdf = 1 + 0.5 * np.sin(4 * np.pi * u)  # non-uniform density
        >>> midpoints, icdf = reconstruct_icdf(u, pdf)
    """
    # Sort samples and corresponding PDF values
    to_sorted_order = np.argsort(u_samples)
    u_sorted = u_samples[to_sorted_order]
    ipdf_sorted = ipdf_values[to_sorted_order]

    # Create midpoints for piecewise linear interpolation
    u_midpoints = (u_sorted[:-1] + u_sorted[1:]) / 2
    u_midpoints = np.array([0] + list(u_midpoints) + [1])

    # Calculate interval widths
    interval_widths = u_midpoints[1:] - u_midpoints[:-1]

    # Integrate PDF to get cumulative ICDF
    cumulative_icdf = np.array([0] + list((interval_widths * ipdf_sorted).cumsum()))

    return u_midpoints, cumulative_icdf


class PositionGraph:
    """
    Circular embedding and position analysis for network graphs.

    This class provides methods to embed a network graph on a circle using
    spring-based forces, analyze the resulting neighborhood structure, and
    reconstruct optimal node positions accounting for local density variations.

    The circular embedding reveals ring-like connectivity patterns in networks
    and enables measurement of "ringity" - how much a network resembles a ring.

    Attributes:
        G (networkx.Graph): Input network graph
        edgelist (list): List of graph edges
        nodelist (list): List of graph nodes
        n_nodes (int): Number of nodes in graph
        pos (ndarray): 2D coordinates from spring embedding
        unadjusted_embedding_dict (dict): Initial angular positions [0, 2π)
        embedding_dict (dict): Recentered and reoriented positions
        rpositions (dict): Final reconstructed positions accounting for density

    Example:
        >>> import networkx as nx
        >>> G = nx.karate_club_graph()
        >>> pg = PositionGraph(G)
        >>> pg.make_circular_spring_embedding()
        >>> pg.process_spring_embedding()
        >>> final_positions = pg.rpositions
    """

    def __init__(self, G):
        """
        Initialize PositionGraph with a NetworkX graph.

        Args:
            G (networkx.Graph): Input graph for analysis
        """
        self.G = nx.relabel_nodes(G, str)
        self.edgelist = list(self.G.edges())
        self.nodelist = sorted(list(self.G.nodes()))
        self.n_nodes = len(self.nodelist)

    def make_circular_spring_embedding(
        self, verbose=False, k=None, iterations=300, pos=None
    ):
        """
        Create circular spring embedding using modified Fruchterman-Reingold algorithm.

        Places nodes on a circle using spring forces while constraining them to
        remain approximately on the unit circle. The algorithm balances attractive
        forces between connected nodes and repulsive forces between all nodes.

        Args:
            verbose (bool): If True, show progress bar and print spring constant
            k (float, optional): Spring constant. If None, uses adaptive value
                                 based on number of nodes: k = 50*2π/n_nodes

        Sets:
            self.pos: 2D coordinates of nodes after spring relaxation
            self.unadjusted_embedding_array: Rank-transformed angular positions
            self.unadjusted_embedding_dict: Node -> angular position mapping

        Notes:
            - Uses 300 iterations with linear cooling schedule
            - Applies radial constraint to keep nodes near unit circle
            - Angular positions are rank-transformed to ensure uniform spacing
        """
        # Convert graph to adjacency matrix
        A = nx.to_numpy_array(self.G, nodelist=self.nodelist, weight="weight")
        dim = 2

        if pos == None:
            # Random initial positions
            pos = 2 * (
                np.asarray(np.random.random((self.n_nodes, dim)), dtype=A.dtype) - 0.5
            )

        # Set spring constant - smaller networks need larger k for proper spacing
        # scaled by linear rather than sqaure root of number of nodes because embedding is 1D
        if k is None:
            k = 50 * (2 * np.pi) / self.n_nodes
            if verbose:
                print(f"Using spring constant k = {k:.4f}")

        # Initial temperature for simulated annealing
        t = max(max(pos.T[0]) - min(pos.T[0]), max(pos.T[1]) - min(pos.T[1])) * 0.1
        dt = t / float(iterations + 1)

        # Displacement array for vectorized force calculation
        delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)

        iterator_ = tqdm.trange(iterations) if verbose else range(iterations)

        for iteration in iterator_:
            if iteration == iterations:
                break

            # Calculate pairwise position differences
            for i in range(pos.shape[1]):
                delta[:, :, i] = pos[:, i, None] - pos[:, i]

            # Pairwise distances with minimum threshold
            distance = np.sqrt((delta**2).sum(axis=-1))
            distance = np.where(distance < 0.01, 0.01, distance)

            # Spring forces: repulsive (k²/d²) and attractive (-A*d/k)
            displacement = np.transpose(
                np.transpose(delta) * (k * k / distance**2 - A * distance / k)
            ).sum(axis=1)

            # Normalize displacement by temperature
            length = np.sqrt((displacement**2).sum(axis=1))
            length = np.where(length < 0.01, 0.01, length)
            delta_pos = np.transpose(np.transpose(displacement) * t / length)

            # Radial constraint: pull nodes back toward unit circle
            radius = np.linalg.norm(pos, axis=1)
            alpha = (iteration / iterations) ** 2  # Increasing constraint strength
            delta_pos += -alpha * pos * (radius - 1)[:, None]

            pos += delta_pos
            t -= dt  # Cool temperature

        self.pos = pos

        self.positions_to_uniform_embedding()

    def load_circular_spring_embedding(self, filename):
        df = pd.read_csv(filename, index_col=0)
        df.index = df.index.astype(str)
        assert set(df.index) == set(self.nodelist)
        self.pos = df.reindex(self.nodelist).values

        self.positions_to_uniform_embedding()

    def positions_to_uniform_embedding(self):
        # Convert to angular positions and rank-transform for uniform spacing
        # unadjusted_embedding_array = np.mod(np.arctan2(pos[:, 0], pos[:, 1]), 2*np.pi)
        self.unadjusted_embedding_array = np.arctan2(self.pos[:, 0], self.pos[:, 1])

        # self.unadjusted_embedding_array = (2 * np.pi *
        #                                  scipy.stats.rankdata(unadjusted_embedding_array) /
        #                                  len(unadjusted_embedding_array))

        self.unadjusted_embedding_dict = dict(
            zip(self.nodelist, self.unadjusted_embedding_array)
        )

        self.nodelist_sorted_unadjusted = sorted(
            self.nodelist, key=self.unadjusted_embedding_dict.get
        )

    def smooth_neighborhood_widths(self, window_size=None):
        """
        Analyze and smooth the neighborhood width around each position on the circle.

        For each node, examines the angular spread of its neighbors to estimate
        local neighborhood width. This reveals regions of high/low connectivity
        density around the circle. The widths are smoothed to reduce noise.

        Sets:
            self.nodelist_sorted_unadjusted: Nodes sorted by angular position
            self.stds_smoothed: Smoothed standard deviations of neighborhood widths
            self.deleteme: Raw neighborhood samples (for debugging)

        Notes:
            - Uses circular statistics to handle wraparound at 0/2π boundary
            - Includes each node in its own neighborhood to avoid empty lists
            - Smoothing window size scales with network size (N/20)
        """

        if window_size == None:
            window_size = int(self.n_nodes / 20)

        neighborhood_samples = []
        for node in self.nodelist_sorted_unadjusted:
            # Get angular positions of neighbors (plus the node itself)
            neighbor_positions = [
                self.unadjusted_embedding_dict[neighbor]
                for neighbor in self.G.neighbors(node)
            ]
            neighbor_positions.append(self.unadjusted_embedding_dict[node])

            # Center positions around circular mean to handle wraparound
            circ_mean = circular_mean(neighbor_positions)
            centered_positions = [
                np.mod(pos - circ_mean + np.pi, 2 * np.pi) for pos in neighbor_positions
            ]

            neighborhood_samples.append(np.array(centered_positions))

        # Smooth the neighborhood width estimates
        self.stds_smoothed = estimate_std(neighborhood_samples, window_size=window_size)

    def recenter_and_reorient_calcs(self):
        """
        Calculate optimal center and orientation for the circular embedding.

        Finds the position where neighborhood density changes most rapidly,
        which typically corresponds to a natural "break" in the ring structure.
        This position becomes the new zero point for the embedding.

        Sets:
            self.look_for_discontinuity_std: Convolution result for finding breaks
            self.embedding_cutoff: Angular position chosen as new zero point
            self.sign_change: Orientation (±1) based on direction of density change

        Notes:
            - Uses edge detection filter to find rapid changes in density
            - Triples the data to handle circular boundary conditions
            - Filter width scales with network size (N/10)
        """
        n_nodes = self.n_nodes

        # Create edge detection filter
        bin_width = round(len(self.nodelist) / 10)
        edge_filter = [-1] * bin_width + [1] * bin_width

        # Apply filter with circular boundary conditions (triple data)
        tripled_stds = (
            list(self.stds_smoothed)
            + list(self.stds_smoothed)
            + list(self.stds_smoothed)
        )

        self.look_for_discontinuity_std = np.convolve(
            tripled_stds, edge_filter, mode="same"
        )

        # Find position of maximum change (in middle third to avoid boundary effects)
        unadjusted_embedding_array = np.array(
            [
                self.unadjusted_embedding_dict[node]
                for node in self.nodelist_sorted_unadjusted
            ]
        )

        max_change_idx = np.argmax(
            np.abs(self.look_for_discontinuity_std[n_nodes : 2 * n_nodes])
        )
        self.embedding_cutoff = unadjusted_embedding_array[max_change_idx]
        self.sign_change = np.sign(
            self.look_for_discontinuity_std[n_nodes : 2 * n_nodes][max_change_idx]
        )

    def reparametrize(self, stds_smoothed):
        """
        Reparametrize positions using inverse probability density function.

        Creates a mapping from uniform angular positions to density-adjusted
        positions by using the reciprocal of neighborhood width as a proxy
        for local density.

        Args:
            stds_smoothed (array): Smoothed standard deviations (currently unused parameter)

        Sets:
            self.nodelist_sorted: Nodes sorted by adjusted embedding positions
            self.ipdf_smooth_adjusted: Inverse PDF values for each position
        """
        # Create inverse PDF from smoothed neighborhood widths
        ipdf_smooth_dict = dict(
            zip(self.nodelist_sorted_unadjusted, self.stds_smoothed)
        )
        self.nodelist_sorted = sorted(self.nodelist, key=self.embedding_dict.get)

        # Use reciprocal of width as density estimate
        self.ipdf_smooth_adjusted = np.array(
            [1 / ipdf_smooth_dict[node] for node in self.nodelist_sorted]
        )

    def recenter_and_reorient(self, reparametrize=True, calculate=True):
        """
        Recenter and reorient the circular embedding based on density analysis.

        Args:
            reparametrize (bool): Whether to compute inverse PDF for final reconstruction
            calculate (bool): Whether to recalculate cutoff and orientation

        Sets:
            self.embedding_dict: Recentered and reoriented angular positions

        Notes:
            - Shifts zero point to the calculated cutoff position
            - May flip orientation based on sign of density change
        """
        if calculate:
            self.recenter_and_reorient_calcs()

        # Apply recentering and reorientation transformation
        self.embedding_dict = {
            node: np.mod(self.sign_change * (pos - self.embedding_cutoff), 2 * np.pi)
            for node, pos in self.unadjusted_embedding_dict.items()
        }

        if reparametrize:
            self.reparametrize(self.stds_smoothed)

    def reconstruct_positions(self):
        """
        Reconstruct final positions using inverse CDF transformation.

        Applies the inverse cumulative distribution function to transform
        from uniform angular spacing to spacing that accounts for local
        density variations. This produces the final "ringity-adjusted" positions.

        Sets:
            self.rpositions: Final reconstructed node positions

        Notes:
            - Uses the inverse PDF computed from neighborhood widths
            - Normalizes final positions to [0, 2π) range
            - These positions are used for final ringity analysis
        """
        # Get positions in sorted order
        embedding_positions = np.array(
            [self.embedding_dict[node] for node in self.nodelist_sorted]
        )

        # Apply inverse CDF transformation
        u_midpoints, cumulative_icdf = reconstruct_icdf(
            embedding_positions / (2 * np.pi), self.ipdf_smooth_adjusted
        )

        # Normalize to [0, 2π) and create final position mapping
        cumulative_icdf = 2 * np.pi * cumulative_icdf / cumulative_icdf[-1]
        self.rpositions = dict(zip(self.nodelist_sorted, cumulative_icdf))

    def process_spring_embedding(self):
        self.smooth_neighborhood_widths()
        self.recenter_and_reorient_calcs()
        self.recenter_and_reorient()
        self.reconstruct_positions()


def plot_edge_positions(ax, graph, node_positions, color="k", point_size=None):
    """
    Scatter plot showing positions of connected node pairs.

    Each point represents a connected pair (i,j) plotted at coordinates
    (position_i, position_j). The pattern reveals the spatial structure
    of connections. Homophilous networks show concentration near the diagonal.

    Parameters
    ----------
    color : str, default 'k'
        Color for the scatter points

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the scatter plot

    Raises
    ------
    ValueError
        If connection model hasn't been fitted
    """

    # Get positions of connected node pairs
    edge_positions = np.array(
        [
            [np.mod(node_positions[i], 2 * np.pi), np.mod(node_positions[j], 2 * np.pi)]
            for i, j in graph.edges()
        ]
    )

    # Add symmetric pairs for visualization
    symmetric_pairs = np.vstack([edge_positions[:, 1], edge_positions[:, 0]]).T
    all_edge_positions = np.vstack([edge_positions, symmetric_pairs])

    # Plot the points
    if point_size == None:
        point_size = 50 / np.sqrt(len(graph.edges()))

    ax.scatter(
        all_edge_positions[:, 0],
        all_edge_positions[:, 1],
        c=color,
        s=point_size,
        alpha=0.6,
    )

    ax.set_xlabel(r"Position $i$", fontsize=12)
    ax.set_ylabel(r"Position $j$", fontsize=12)
    ax.set_title(r"Connected Node Pairs $(i,j)$")

    # Set circular ticks
    tick_positions = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    tick_labels = ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    # Draw interaction width boundaries
    # ax.plot([0, 2*np.pi], [self.interaction_width, 2*np.pi + self.interaction_width],
    #        c="#777777", linewidth=2, linestyle='--', alpha=0.8,
    #        label=f'Interaction width = {self.interaction_width:.3f}')
    # ax.plot([0, 2*np.pi], [-self.interaction_width, 2*np.pi - self.interaction_width],
    #        c="#777777", linewidth=2, linestyle='--', alpha=0.8)

    ax.set_xlim([0, 2 * np.pi])
    ax.set_ylim([0, 2 * np.pi])
    ax.legend()
    ax.grid(True, alpha=0.3)


def load_network(filename):
    G = nx.read_gml(filename)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.relabel_nodes(G, str)
    return G


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-filename", type=str, default="ws_graph.gml")
    parser.add_argument("--input-true-positions", type=str, default="ws_positions.csv")
    parser.add_argument("--output-true-comparison-plot", type=str, default="none")
    parser.add_argument("--output-folder", type=str, default="none")
    parser.add_argument("--iterations", type=int, default=500)

    args = parser.parse_args()

    input_filename = args.input_filename
    output_folder = Path(args.output_folder)
    iterations = args.iterations

    os.makedirs(output_folder, exist_ok=True)

    G = load_network(input_filename)

    self = PositionGraph(G)

    self.make_circular_spring_embedding(verbose=True, iterations=iterations)

    self.process_spring_embedding()

    if args.output_folder != "none":
        df = pd.DataFrame(self.pos, index=self.nodelist)
        df.to_csv(output_folder / f"positions_spring.csv")

    positions_df = pd.Series(self.rpositions, index=self.nodelist)
    positions_df.to_csv(output_folder / f"positions.csv")

    if args.input_true_positions != "none":
        true_positions_df = pd.read_csv(args.input_true_positions)["0"]

        positions_df.index = positions_df.index.astype(str)
        true_positions_df.index = true_positions_df.index.astype(str)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.scatter(
            true_positions_df.reindex(positions_df.index).values,
            positions_df.values,
            rasterized=True,
            c="k",
        )

        tick_positions = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        tick_labels = [
            "0",
            r"$\frac{\pi}{2}$",
            r"$\pi$",
            r"$\frac{3\pi}{2}$",
            r"$2\pi$",
        ]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

        ax.set_xlabel("true positions")
        ax.set_ylabel("reconstructed positions")

        if args.output_true_comparison_plot != "none":

            fig.savefig(args.output_true_comparison_plot)

        else:
            fig.show()
            plt.show()
