import os
import fcntl
import numpy as np
import networkx as nx
import scipy.optimize
import itertools as it
import matplotlib.pyplot as plt

import ringity as rng

rng.set_theme()

from pathlib import Path
from retrieve_positions import PositionGraph

cmap = {
    "immune.gml": "#2D4452",
    "fibro.gml": "#20313C",
    "soil.gml": "#132027",
    "lipid.gml": "#142935",
    "gene.gml": "#102C3D",
}


def truncated_exponential(theta, scale, decay_rate, truncation_point):
    """
    Truncated exponential distribution for circular/angular data.

    Parameters
    ----------
    theta : array-like
        Angular positions (0 to 2π)
    scale : float
        Scaling parameter (controls overall magnitude)
    decay_rate : float
        Exponential decay parameter (controls how quickly probability decreases)
    truncation_point : float
        Initial/truncation point (flexible starting position)

    Returns
    -------
    array-like
        Probability density values
    """
    return (
        scale
        * decay_rate
        * np.exp(decay_rate * np.mod(-(theta - truncation_point), 2 * np.pi))
    )


def piecewise_linear_decay(theta, max_probability, interaction_width):
    """
    Piecewise linear decay function for connection probability vs distance.

    Models the probability that two nodes are connected as a function of their
    angular distance. Probability starts at max_probability for nodes at the
    same position and linearly decreases to zero at interaction_width.

    Parameters
    ----------
    theta : array-like
        Angular distances between node pairs
    max_probability : float
        Maximum connection probability (for nodes at same position)
    interaction_width : float
        Distance at which connection probability becomes zero

    Returns
    -------
    array-like
        Connection probabilities
    """
    temp = interaction_width - theta
    return max_probability * 0.5 * (np.abs(temp) + temp)


class NetworkHomophilyAnalyzer:
    """
    Analyzer for testing homophily/assortativity in networks with circular node positions.

    This class tests whether networks exhibit homophily (preference for connecting
    to similar/nearby nodes) by analyzing the relationship between connection
    probability and angular distance on a circle.

    A network is considered homophilous/assortative when nodes preferentially
    connect to other nodes that are close to them in the circular space.

    Typical Usage
    -------------
    analyzer = NetworkHomophilyAnalyzer(graph, node_positions)

    # Generate analysis plots
    fig1 = analyzer.plot_distance_histogram()
    analyzer.fit_connection_probability_model()
    fig2 = analyzer.plot_connection_probability()
    fig3 = analyzer.plot_edge_positions()

    # Check if network is homophilous
    if analyzer.interaction_width < threshold:  # Small width indicates homophily
        print("Network shows homophilous behavior")

    Parameters
    ----------
    graph : networkx.Graph
        The network to analyze
    node_positions : dict
        Dictionary mapping node IDs to angular positions (0 to 2π)
    """

    def __init__(self, graph, node_positions):
        """
        Initialize the analyzer.

        Parameters
        ----------
        graph : networkx.Graph
            Network to analyze
        node_positions : dict
            Node positions as {node_id: angular_position} where positions are in [0, 2π]
        """
        self.graph = graph
        self.nodelist = list(graph.nodes())
        self.node_positions = node_positions

        # Will be populated by analysis methods
        self.distance_bins = None
        self.distance_midpoints = None
        self.neighbor_counts = None
        self.total_counts = None
        self.connection_probabilities = None

        # Fitted parameters
        self.max_probability = None
        self.interaction_width = None

    def _calculate_angular_distance(self, pos1, pos2):
        """Calculate shortest angular distance between two positions on a circle."""
        euclidean_dist = np.abs(pos2 - pos1)
        return np.min([euclidean_dist, 2 * np.pi - euclidean_dist])

    def _calculate_angular_mean(self, pos1, pos2):
        """Calculate angular mean between two positions on a circle."""
        circular_mean_x = np.cos(pos1) + np.cos(pos2)
        circular_mean_y = np.sin(pos1) + np.sin(pos2)

        angular_mean = np.arctan2(circular_mean_x, circular_mean_y)

        return np.min([angular_mean, 2 * np.pi - angular_mean])

    def plot_distance_histogram(self, color="k", n_bins=50):
        """
        Plot histogram of distances for all node pairs vs. connected pairs.

        This visualization shows how connection frequency varies with distance.
        If the network is homophilous, we expect to see more connections at
        shorter distances compared to the overall distribution.

        Parameters
        ----------
        color : str, default 'k'
            Color for the connected pairs histogram
        n_bins : int, default 50
            Number of histogram bins

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the histogram plot
        """
        # Calculate distances for connected node pairs
        connected_distances = []
        for node_i in self.nodelist:
            for node_j in self.graph.neighbors(node_i):
                dist = self._calculate_angular_distance(
                    self.node_positions[node_j], self.node_positions[node_i]
                )
                connected_distances.append(dist)

        # Calculate distances for all possible node pairs
        all_distances = []
        for node_i, node_j in it.product(self.nodelist, repeat=2):
            dist = self._calculate_angular_distance(
                self.node_positions[node_j], self.node_positions[node_i]
            )
            all_distances.append(dist)

        # Create histogram
        fig, ax = plt.subplots(figsize=(10, 6))

        self.total_counts, bins, _ = ax.hist(
            all_distances,
            bins=n_bins,
            color="#AAAAAA",
            density=False,
            label="All pairs",
            alpha=0.7,
        )
        self.neighbor_counts, _, _ = ax.hist(
            connected_distances, bins=bins, color=color, label="Connected pairs"
        )

        # Store for later analysis
        self.distance_bins = bins
        self.distance_midpoints = (bins[1:] + bins[:-1]) / 2
        self.connection_probabilities = self.neighbor_counts / self.total_counts

        ax.set_xlabel("Angular Distance", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title("Distribution of Distances: All Pairs vs Connected Pairs")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def _create_embedding_sections(self, n_embedding_sections):
        self.n_embedding_sections = n_embedding_sections
        self.embedding_section_boundaries = np.linspace(
            -np.pi, np.pi, n_embedding_sections + 1
        )

    def _get_section_index(self, x):

        section_index_incremented = np.digitize(
            np.mod(x, 2 * np.pi) - np.pi, self.embedding_section_boundaries
        )
        assert section_index_incremented > 0
        return section_index_incremented - 1

    def calculate_distance_histogram_by_section(
        self,
        color="k",
        n_bins=50,
        n_embedding_sections=5,
    ):
        """
        Calculate histogram of distances for all node pairs vs. connected pairs,
        split by sections of the embdding space. This will indicate if interaction
        width varies widely across position space.


        Parameters
        ----------
        color : str, default 'k'
            Color for the connected pairs histogram
        n_bins : int, default 50
            Number of histogram bins
        n_embedding_sections : int, default 5


        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the histogram plot

        """

        self._create_embedding_sections(n_embedding_sections)

        # Calculate distances for connected node pairs
        connected_distances = [[] for _ in range(n_embedding_sections)]
        for node_i in self.nodelist:

            for node_j in self.graph.neighbors(node_i):

                angular_mean = self._calculate_angular_mean(
                    self.node_positions[node_j], self.node_positions[node_i]
                )

                section_index = self._get_section_index(angular_mean)  #

                dist = self._calculate_angular_distance(
                    self.node_positions[node_j], self.node_positions[node_i]
                )
                connected_distances[section_index].append(dist)

        # Calculate distances for all possible node pairs
        all_distances = [[] for _ in range(n_embedding_sections)]
        for node_i, node_j in it.product(self.nodelist, repeat=2):

            angular_mean = self._calculate_angular_mean(
                self.node_positions[node_j], self.node_positions[node_i]
            )

            section_index = self._get_section_index(angular_mean)

            dist = self._calculate_angular_distance(
                self.node_positions[node_j], self.node_positions[node_i]
            )
            all_distances[section_index].append(dist)

        self.total_counts_split = [None for _ in range(n_embedding_sections)]
        self.neighbor_counts_split = [None for _ in range(n_embedding_sections)]
        self.connection_probabilities_split = [
            None for _ in range(n_embedding_sections)
        ]
        self.distance_bins_split = [None for _ in range(n_embedding_sections)]
        self.distance_midpoints_split = [None for _ in range(n_embedding_sections)]

        for section_index in range(n_embedding_sections):
            # Create histogram
            fig, ax = plt.subplots(figsize=(10, 6))

            self.total_counts_split[section_index], bins, _ = ax.hist(
                all_distances[section_index],
                bins=n_bins,
                color="#AAAAAA",
                density=False,
                label="All pairs",
                alpha=0.7,
            )
            self.neighbor_counts_split[section_index], _, _ = ax.hist(
                connected_distances[section_index],
                bins=bins,
                color=color,
                label="Connected pairs",
            )
            ax.set_xlabel("Angular Distance", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(
                f"""Distribution of Distances: All Pairs vs Connected Pairs\n
                            for {self.embedding_section_boundaries[section_index]} to {self.embedding_section_boundaries[section_index+1]}"""
            )
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Store for later analysis
            self.distance_bins_split[section_index] = bins
            self.distance_midpoints_split[section_index] = (bins[1:] + bins[:-1]) / 2

            self.connection_probabilities_split[section_index] = (
                self.neighbor_counts_split[section_index]
                / self.total_counts_split[section_index]
            )

            plt.close(fig)

    def fit_connection_probability_model(self):
        """
        Fit piecewise linear model to connection probability vs distance.

        This method fits a model where connection probability decreases linearly
        from a maximum value at distance 0 to zero at the interaction width.

        Lower interaction_width values indicate stronger homophily/assortativity.

        Raises
        ------
        ValueError
            If plot_distance_histogram() hasn't been called first
        """
        if self.connection_probabilities is None:
            raise ValueError(
                "Must call plot_distance_histogram() first to generate data"
            )

        # Fit model (skip first bin to avoid self-loops)
        valid_indices = slice(1, None)
        typical_scale = self.connection_probabilities[valid_indices].mean()
        fitted_params, _ = scipy.optimize.curve_fit(
            piecewise_linear_decay,
            self.distance_midpoints[valid_indices],
            self.connection_probabilities[valid_indices] / typical_scale,
            p0=None,
            sigma=None,
        )

        max_probability_unscaled, self.interaction_width = fitted_params

        self.max_probability = max_probability_unscaled * typical_scale

    def plot_connection_probability(self, color="k"):
        """
        Plot connection probability vs distance with fitted model.

        Shows the empirical connection probability (proportion of node pairs
        that are connected) as a function of angular distance, along with
        the fitted piecewise linear model.

        Parameters
        ----------
        color : str, default 'k'
            Color for the empirical data line

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the probability plot

        Raises
        ------
        ValueError
            If required analysis hasn't been performed first
        """
        if self.connection_probabilities is None:
            raise ValueError("Must call plot_distance_histogram() first")
        if self.max_probability is None:
            raise ValueError("Must call fit_connection_probability_model() first")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot empirical probabilities
        ax.plot(
            self.distance_midpoints,
            self.connection_probabilities,
            color=color,
            linewidth=3,
            label="Empirical",
            marker="o",
            markersize=4,
        )

        # Plot fitted model
        theta_space = np.linspace(0, np.pi, 100)
        fitted_probs = piecewise_linear_decay(
            theta_space, self.max_probability, self.interaction_width
        )
        ax.plot(
            theta_space,
            fitted_probs,
            color="#777777",
            linewidth=3,
            label=f"Fitted Model (width={self.interaction_width:.3f})",
            linestyle="--",
        )

        ax.set_ylim((0, None))
        ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
        ax.set_xticklabels(
            ["0", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"]
        )
        ax.set_xlabel("Angular Distance", fontsize=12)
        ax.set_ylabel("Connection Probability", fontsize=12)
        ax.set_title("Connection Probability vs Distance")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def plot_connection_probability_by_section(self, color="k"):
        """
        Plot connection probability vs distance with fitted model for nodes in a given section.

        Shows the empirical connection probability (proportion of node pairs
        that are connected) as a function of angular distance, along with
        the fitted piecewise linear model.

        Parameters
        ----------
        color : str, default 'k'
            Color for the empirical data line

        Returns
        -------
        matplotlib.figure.Figure
            Figure containing the probability plot

        Raises
        ------
        ValueError
            If required analysis hasn't been performed first
        """

        if self.connection_probabilities_split is None:
            raise ValueError("Must call plot_distance_histogram_by_section() first")

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [
            np.array([1.0, 1.0, 1.0]) * i / self.n_embedding_sections
            for i in range(self.n_embedding_sections)
        ]
        m_list = []
        for section_index, color in zip(range(self.n_embedding_sections), colors):

            # Plot empirical probabilities
            m = ax.plot(
                self.distance_midpoints_split[section_index],
                self.connection_probabilities_split[section_index],
                color=color,
                linewidth=3,
                label="Empirical",
                marker="o",
                markersize=4,
            )
            m_list.append(m)

        ax.set_ylim((0, None))
        ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
        ax.set_xticklabels(
            ["0", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"]
        )
        ax.set_xlabel("Angular Distance", fontsize=12)
        ax.set_ylabel("Connection Probability", fontsize=12)
        ax.set_title("Connection Probability vs Distance")
        ax.legend(
            m_list,
            labels=[
                f"{self.embedding_section_boundaries[section_index]:.3f} to {self.embedding_section_boundaries[section_index+1]:.3f}"
                for section_index in range(self.n_embedding_sections)
            ],
        )
        ax.grid(True, alpha=0.3)

        return fig

    def plot_edge_positions(self, color="k"):
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
        if self.interaction_width is None:
            raise ValueError("Must call fit_connection_probability_model() first")

        fig, ax = plt.subplots(figsize=(8, 8))

        # Get positions of connected node pairs
        edge_positions = np.array(
            [
                [self.node_positions[i], self.node_positions[j]]
                for i, j in self.graph.edges()
            ]
        )

        # Add symmetric pairs for visualization
        symmetric_pairs = np.vstack([edge_positions[:, 1], edge_positions[:, 0]]).T
        all_edge_positions = np.vstack([edge_positions, symmetric_pairs])

        # Plot the points
        point_size = max(1, 50 / np.sqrt(len(self.graph.edges())))
        ax.scatter(
            all_edge_positions[:, 0],
            all_edge_positions[:, 1],
            c=color,
            s=point_size,
            alpha=0.6,
            rasterized=True,
        )

        ax.set_xlabel(r"Position $i$", fontsize=12)
        ax.set_ylabel(r"Position $j$", fontsize=12)
        ax.set_title(r"Connected Node Pairs $(i,j)$")

        # Set circular ticks
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

        # Draw interaction width boundaries
        ax.plot(
            [0, 2 * np.pi],
            [self.interaction_width, 2 * np.pi + self.interaction_width],
            c="#777777",
            linewidth=2,
            linestyle="--",
            alpha=0.8,
            label=f"Interaction width = {self.interaction_width:.3f}",
        )
        ax.plot(
            [0, 2 * np.pi],
            [-self.interaction_width, 2 * np.pi - self.interaction_width],
            c="#777777",
            linewidth=2,
            linestyle="--",
            alpha=0.8,
        )

        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 2 * np.pi])
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def _plot_edge_positions_with_section(self):
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

        fig, ax = plt.subplots(figsize=(8, 8))

        # Get positions of connected node pairs
        edge_positions = np.array(
            [
                [self.node_positions[i], self.node_positions[j]]
                for i, j in self.graph.edges()
            ]
        )

        # Add symmetric pairs for visualization
        symmetric_pairs = np.vstack([edge_positions[:, 1], edge_positions[:, 0]]).T
        all_edge_positions = np.vstack([edge_positions, symmetric_pairs])

        circular_means = [
            self._calculate_angular_mean(
                all_edge_positions[i, 0], all_edge_positions[i, 1]
            )
            for i in range(all_edge_positions.shape[0])
        ]

        color = [
            self._get_section_index(circular_mean) for circular_mean in circular_means
        ]
        # Plot the points
        point_size = max(1, 50 / np.sqrt(len(self.graph.edges())))
        ax.scatter(
            all_edge_positions[:, 0],
            all_edge_positions[:, 1],
            c=color,
            s=point_size,
            alpha=0.6,
            rasterized=True,
        )

        ax.set_xlabel(r"Position $i$", fontsize=12)
        ax.set_ylabel(r"Position $j$", fontsize=12)
        ax.set_title(r"Connected Node Pairs $(i,j)$")

        # Set circular ticks
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

        ax.set_xlim([0, 2 * np.pi])
        ax.set_ylim([0, 2 * np.pi])
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig

    def get_homophily_summary(self):
        """
        Get summary statistics for homophily analysis.

        Returns
        -------
        dict
            Dictionary containing fitted parameters and interpretation
        """
        if self.max_probability is None:
            raise ValueError("Must run analysis first")

        return {
            "max_probability": self.max_probability,
            "interaction_width": self.interaction_width,
            "is_homophilous": self.interaction_width < np.pi / 2,  # Heuristic threshold
            "homophily_strength": (
                "Strong"
                if self.interaction_width < np.pi / 4
                else "Moderate" if self.interaction_width < np.pi / 2 else "Weak"
            ),
        }


def analyze_position_distribution(n_nodes, node_positions, n_bins=None, color="k"):
    """
    Analyze and fit the distribution of node positions on the circle.

    This function tests the hypothesis that node positions follow a truncated
    exponential distribution, which is commonly observed in spatial networks.

    Parameters
    ----------
    n_nodes : int
        Number of nodes
    node_positions : dict
        Dictionary mapping node IDs to angular positions
    n_bins : int, optional
        Number of histogram bins (default: sqrt(n_nodes))
    color : str, default 'k'
        Color for histogram bars

    Returns
    -------
    tuple
        (figure, scale, decay_rate, truncation_point) where the last three
        are the fitted parameters of the truncated exponential
    """
    if n_bins is None:
        n_bins = int(np.sqrt(n_nodes))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram of positions
    positions = list(node_positions.values())
    counts, bins, _ = ax.hist(
        positions, bins=n_bins, color=color, density=True, alpha=0.7, label="Empirical"
    )

    ax.set_xlabel("Angular Position", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Distribution of Node Positions")

    # Set circular ticks
    tick_positions = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
    tick_labels = ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    # Fit truncated exponential
    midpoints = (bins[1:] + bins[:-1]) / 2
    fitted_params, _ = scipy.optimize.curve_fit(
        truncated_exponential, midpoints, counts, p0=None, sigma=None
    )

    scale, decay_rate, truncation_point = fitted_params

    # Plot fitted curve
    theta_space = np.linspace(0, 2 * np.pi, 100)
    fitted_density = truncated_exponential(
        theta_space, scale, decay_rate, truncation_point
    )
    ax.plot(
        theta_space,
        fitted_density,
        color="#777777",
        linewidth=3,
        label=f"Fitted Truncated Exponential",
        linestyle="--",
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, scale, decay_rate, truncation_point


# Example usage and testing functions
def create_test_network(n_nodes=100, k=6, p=0.1, seed=42):
    """
    Create a test network using Watts-Strogatz model with circular positions.

    Parameters
    ----------
    n_nodes : int
        Number of nodes
    k : int
        Each node is joined with its k nearest neighbors in a ring topology
    p : float
        Probability of rewiring each edge
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (networkx.Graph, dict) - the graph and node positions
    """
    np.random.seed(seed)

    # Create Watts-Strogatz graph
    graph = nx.watts_strogatz_graph(n_nodes, k, p, seed=seed)

    # Assign circular positions
    positions = {}
    for i, node in enumerate(graph.nodes()):
        positions[node] = (2 * np.pi * i) / n_nodes

    return graph, positions


def load_network(filename):
    G = nx.read_gml(filename)
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.relabel_nodes(G, str)
    return G


import pandas as pd


def run_analysis(
    network_file="data/empirical_networks/lipid.gml",
    embedding_file="data/positions/lipid/positions.csv",
    load_embedding=True,
    randomization="none",
    figure_output_folder="test/",
    embedding_iterations=10,
    draw_figures=True,
    display_figures=False,
    summary_info_file=None,
    append_summary=False,
):

    from pathlib import Path

    figure_output_folder = Path(figure_output_folder)

    graph = load_network(network_file)

    color = cmap[os.path.split(network_file)[1]]

    if randomization == "configuration":
        graph = nx.configuration_model(list(dict(graph.degree()).values()))
        graph = nx.relabel_nodes(graph, str)

    elif randomization in ["none", "None", "False", "false", "no-randomization"]:
        randomization = False
    else:
        print(
            f"randomization method {randomization} not found,\ndefaulting to no randomization..."
        )
        randomization = False

    if load_embedding:

        assert not randomization

        positions = pd.read_csv(embedding_file, index_col=0)["0"]
        positions.index = positions.index.astype(str)
        positions = positions.to_dict()

        embedding_iterations = -1

    else:
        graph_position_finder = PositionGraph(graph)
        graph_position_finder.make_circular_spring_embedding(
            iterations=embedding_iterations
        )
        graph_position_finder.process_spring_embedding()
        positions = graph_position_finder.rpositions

    analyzer = NetworkHomophilyAnalyzer(graph, positions)

    fig_distance_histogram = analyzer.plot_distance_histogram()
    fig_distance_histogram.suptitle("Step 1: Distance Distribution Analysis")

    analyzer.fit_connection_probability_model()
    analyzer.calculate_distance_histogram_by_section()
    analyzer._plot_edge_positions_with_section()

    if draw_figures or display_figures:

        fig_connection_probability_by_section = (
            analyzer.plot_connection_probability_by_section(color=color)
        )
        fig_connection_probability_by_section.show()

        fig_connection_probability = analyzer.plot_connection_probability(color=color)
        fig_connection_probability.suptitle(
            "Step 2: Connection Probability vs Distance"
        )

        fig_plot_edge_positions = analyzer.plot_edge_positions(color=color)
        fig_plot_edge_positions.suptitle("Step 3: Spatial Pattern of Connections")

        # Analyze position distribution
        (
            fig_position_distribution,
            scale,
            decay_rate,
            trunc_point,
        ) = analyze_position_distribution(len(graph.nodes()), positions, color=color)
        fig_position_distribution.suptitle(
            "Step 4: Node Position Distribution Analysis"
        )

        print(f"\nPosition Distribution Parameters:")
        print(f"Scale: {scale:.3f}")
        print(f"Decay rate: {decay_rate:.3f}")
        print(f"Truncation point: {trunc_point:.3f}")

    if display_figures:

        fig_distance_histogram.show()
        plt.show()

        fig_connection_probability_by_section.show()
        plt.show()

        fig_connection_probability.show()
        plt.show()

        fig_plot_edge_positions.show()
        plt.show()

        fig_position_distribution.show()
        plt.show()

    if draw_figures:

        os.makedirs(figure_output_folder, exist_ok=True)
        fig_distance_histogram.savefig(figure_output_folder / "distance_histogram.pdf")
        fig_connection_probability_by_section.savefig(
            figure_output_folder / "connection_probability_by_section.pdf"
        )
        fig_connection_probability.savefig(
            figure_output_folder / "connection_probability.pdf"
        )

        fig_plot_edge_positions.savefig(
            figure_output_folder / "plot_edge_positions.pdf"
        )
        fig_position_distribution.savefig(
            figure_output_folder / "position_distribution.pdf"
        )

        print("figures saved to ", figure_output_folder)

    if append_summary:

        info = pd.DataFrame(
            [
                [
                    network_file,
                    embedding_file,
                    embedding_iterations,
                    randomization,
                    analyzer.interaction_width,
                    analyzer.max_probability,
                ]
            ]
        )
        info.columns = [
            "network_file",
            "embedding_file",
            "embedding_iterations",
            "randomization",
            "analyzer_interaction_width",
            "analyzer_max_probability",
        ]

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


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--network-file", type=str, default="data/empirical_networks/immune.gml"
    )
    parser.add_argument("--embedding-file", type=str, default="none")
    parser.add_argument("--randomization", type=str, default="none")
    parser.add_argument("--summary-output-file", type=str, default="none")
    parser.add_argument("--figure-output-folder", type=str, default="none")
    parser.add_argument("--embedding-iterations", type=int, default=100)
    parser.add_argument("--display-figures", type=bool, default=False)

    args = parser.parse_args()

    load_embedding = args.embedding_file != "none"
    append_summary = args.summary_output_file != "none"
    draw_figures = args.figure_output_folder != "none"

    run_analysis(
        network_file=args.network_file,
        embedding_file=args.embedding_file,
        load_embedding=load_embedding,
        randomization=args.randomization,
        figure_output_folder=args.figure_output_folder,
        embedding_iterations=args.embedding_iterations,
        draw_figures=draw_figures,
        display_figures=args.display_figures,
        summary_info_file=args.summary_output_file,
        append_summary=append_summary,
    )
