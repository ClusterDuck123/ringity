#!/usr/bin/env python3
"""
Network Homophily Analysis - Complete Analysis Script

This script provides a comprehensive framework for analyzing homophily patterns in networks
using circular spring embeddings and spatial analysis techniques.

Features:
- Network loading and preprocessing
- Circular spring embedding generation
- Homophily analysis with multiple visualization methods
- Control group comparison using configuration models
- Automated figure saving to output directory

Author: Joel Hancock
"""

# Standard library imports
import os
import sys
import json
import uuid
import argparse
from pathlib import Path

# Scientific computing imports
import numpy as np
import scipy.optimize
import scipy.stats
import pandas as pd

# Network analysis imports
import networkx as nx
import ringity as rg

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# Progress tracking
import tqdm

# Custom module imports
from retrieve_positions import PositionGraph
from fitting_model import NetworkHomophilyAnalyzer, analyze_position_distribution

# Configuration constants
EMPIRICALNET_DIR = Path("../data/empirical_networks")
OUTPUT_DIR = Path("output")
POSITIONS_DIR = Path("data/empirical_networks/positions")


def setup_output_directory():
    """Create output directory for saving figures and results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory created: {OUTPUT_DIR}")
    return OUTPUT_DIR


def load_network(name):
    """
    Load a network from GML file and preprocess it.
    
    Args:
        name (str): Name of the network file (without .gml extension)
    
    Returns:
        networkx.Graph: Preprocessed network graph
    """
    print(f"Loading network: {name}")
    G = nx.read_gml(EMPIRICALNET_DIR / f"{name}.gml")
    
    # Remove self-loops and ensure string node labels
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.relabel_nodes(G, str)
    
    print(f"Network loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")
    return G


def run_analysis(G, verbose=True, n_fitting_iters=50):
    """
    Run complete homophily analysis on a network.
    
    Args:
        G (networkx.Graph): Input network
        verbose (bool): Whether to print verbose output
        n_fitting_iters (int): Number of iterations for spring embedding
    
    Returns:
        tuple: (PositionGraph object, NetworkHomophilyAnalyzer object)
    """
    print("Starting network analysis...")
    
    # Create position graph and generate circular spring embedding
    position_graph = PositionGraph(G)
    position_graph.make_circular_spring_embedding(verbose=verbose, iterations=n_fitting_iters)
    position_graph.process_spring_embedding()
    
    # Initialize homophily analyzer
    analyzer = NetworkHomophilyAnalyzer(position_graph.G, position_graph.rpositions)
    
    # Generate distance histogram plot
    analyzer.plot_distance_histogram()
    
    print("Analysis completed successfully")
    return position_graph, analyzer


def calculate_homophily_score(G, n_fitting_iters=50):
    """
    Calculate homophily score for a given network.
    
    Args:
        G (networkx.Graph): Input network
        n_fitting_iters (int): Number of iterations for spring embedding
    
    Returns:
        float: Interaction width as homophily score
    """
    _, analyzer = run_analysis(G, verbose=False, n_fitting_iters=n_fitting_iters)
    return analyzer.interaction_width


def save_control_homophily(network_name="fibro", output_file="test.json", 
                          n_control_iters=100, n_fitting_iters=50):
    """
    Generate control homophily scores using configuration model.
    
    Args:
        network_name (str): Name of the empirical network
        output_file (str): Output JSON file path
        n_control_iters (int): Number of control iterations
        n_fitting_iters (int): Number of fitting iterations per network
    """
    print(f"Generating control homophily scores for {network_name}...")
    
    # Load true network
    G_true = load_network(network_name)
    
    # Generate control networks and calculate homophily scores
    control_homophily = []
    for i in tqdm.trange(n_control_iters, desc="Control iterations"):
        # Create configuration model with same degree sequence
        G_config = nx.configuration_model(list(dict(G_true.degree()).values()))
        homophily_score = calculate_homophily_score(G_config, n_fitting_iters=n_fitting_iters)
        control_homophily.append(homophily_score)
    
    # Calculate true homophily score
    true_homophily = calculate_homophily_score(G_true, n_fitting_iters=n_fitting_iters)
    
    # Save results
    info = {
        "true_homophily": true_homophily, 
        "control_homophily": control_homophily,
        "network_name": network_name,
        "n_control_iters": n_control_iters,
        "n_fitting_iters": n_fitting_iters
    }
    
    with open(output_file, "w") as fp:
        json.dump(info, fp, indent=2)
    
    print(f"Control homophily data saved to {output_file}")


def load_and_plot(file="test.json", downsample_to=None):
    """
    Load homophily data and create comparison plot.
    
    Args:
        file (str): Path to JSON file with homophily data
        downsample_to (int, optional): Number of control samples to use
    """
    print(f"Loading homophily data from {file}")
    
    with open(file) as fp:
        info = json.load(fp)
    
    control_homophily = info["control_homophily"]
    
    # Downsample if requested
    if downsample_to is not None:
        assert downsample_to < len(control_homophily)
        control_homophily = np.random.choice(
            control_homophily, 
            size=(downsample_to,), 
            replace=False
        )
    
    plot_homophily_comparison(control_homophily, info["true_homophily"])


def plot_homophily_comparison(control_homophily, true_homophily, save_fig=True):
    """
    Create violin plot comparing control and true homophily scores.
    
    Args:
        control_homophily (list): Control homophily scores
        true_homophily (float): True network homophily score
        save_fig (bool): Whether to save the figure
    """
    print("Creating homophily comparison plot...")
    
    # Perform statistical test
    stat_result = scipy.stats.wilcoxon(control_homophily, [true_homophily])
    print(f"Wilcoxon test result: {stat_result}")
    
    # Create violin plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=[control_homophily, [true_homophily]], 
        ax=ax
    )
    
    ax.set_xticklabels(['Control Networks', 'True Network'])
    ax.set_ylabel('Homophily Score (Interaction Width)')
    ax.set_title('Homophily Score Comparison: True vs Control Networks')
    
    # Add statistical test result to plot
    ax.text(0.02, 0.98, f'Wilcoxon p-value: {stat_result.pvalue:.3e}', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if save_fig:
        fig_path = OUTPUT_DIR / "homophily_comparison.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {fig_path}")
    
    plt.show()
    return fig



def run_analysis(network_name = "soil",load_positions=False):
    """Run a complete example analysis on a test network."""
    
    # Setup output directory
    setup_output_directory()
    
    # Network selection
    
    print(f"Analyzing network: {network_name}")
    
    # Load network
    graph = load_network(network_name)
    
    # Load or generate positions
    graph_position_finder = PositionGraph(graph)
    
    position_file = POSITIONS_DIR / f"{network_name}.csv"
    if position_file.exists() and load_positions:
        print(f"Loading pre-computed positions from {position_file}")
        graph_position_finder.load_circular_spring_embedding(str(position_file))
    else:
        print("Computing circular spring embedding...")
        iterations = 50
        graph_position_finder.make_circular_spring_embedding(verbose=True, iterations=iterations)
        
        # Save positions for future use
        os.makedirs(POSITIONS_DIR, exist_ok=True)
        df = pd.DataFrame(graph_position_finder.pos, index=graph_position_finder.nodelist)
        df.to_csv(position_file)
        print(f"Positions saved to {position_file}")
    
    # Process spring embedding
    graph_position_finder.process_spring_embedding()
    positions = graph_position_finder.rpositions
    
    # Initialize analyzer
    print("Running homophily analysis...")
    analyzer = NetworkHomophilyAnalyzer(graph, positions)
    
    # Step 1: Distance histogram
    print("Step 1: Generating distance histogram...")
    fig1 = analyzer.plot_distance_histogram()
    fig1.suptitle("Step 1: Distance Distribution Analysis")
    fig1.savefig(OUTPUT_DIR / "step1_distance_histogram.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved: {OUTPUT_DIR / 'step1_distance_histogram.png'}")
    
    # Section-based analysis
    analyzer.calculate_distance_histogram_by_section()
    analyzer._plot_edge_positions_with_section()
    
    fig_sections = analyzer.plot_connection_probability_by_section()
    fig_sections.savefig(OUTPUT_DIR / "connection_probability_by_section.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved: {OUTPUT_DIR / 'connection_probability_by_section.png'}")
    
    # Step 2: Connection probability model
    print("Step 2: Fitting connection probability model...")
    analyzer.fit_connection_probability_model()
    
    fig2 = analyzer.plot_connection_probability()
    fig2.suptitle("Step 2: Connection Probability vs Distance")
    fig2.savefig(OUTPUT_DIR / "step2_connection_probability.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved: {OUTPUT_DIR / 'step2_connection_probability.png'}")
    
    # Step 3: Edge positions
    print("Step 3: Plotting spatial pattern of connections...")
    fig3 = analyzer.plot_edge_positions()
    fig3.suptitle("Step 3: Spatial Pattern of Connections")
    fig3.savefig(OUTPUT_DIR / "step3_edge_positions.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved: {OUTPUT_DIR / 'step3_edge_positions.png'}")
    
    # Print summary
    summary = analyzer.get_homophily_summary()
    print("\n" + "="*40)
    print("HOMOPHILY ANALYSIS RESULTS")
    print("="*40)
    print(f"Maximum connection probability: {summary['max_probability']:.3f}")
    print(f"Interaction width: {summary['interaction_width']:.3f}")
    
    # Step 4: Position distribution analysis
    print("Step 4: Analyzing node position distribution...")
    fig4, scale, decay_rate, trunc_point = analyze_position_distribution(
        len(graph.nodes()), positions
    )
    fig4.suptitle("Step 4: Node Position Distribution Analysis")
    
    print(f"\nPosition Distribution Parameters:")
    print(f"Scale: {scale:.3f}")
    print(f"Decay rate: {decay_rate:.3f}")
    print(f"Truncation point: {trunc_point:.3f}")
    
    # Show all plots
    plt.show()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - All figures saved to output/ directory")
    print("="*60)
    
    return analyzer, (fig1, fig2, fig3, fig4)


def run_control_analysis():
    """Run control analysis for multiple networks."""
    print("Running control analysis for multiple networks...")
    
    # Setup directories
    setup_output_directory()
    folder = Path("saved_control_homophily_scores")
    os.makedirs(folder, exist_ok=True)
    
    # Networks to analyze
    networks = ["lipid", "fibro", "immune", "soil", "gene"]
    
    for network in networks:
        print(f"\nProcessing network: {network}")
        unique_id = str(uuid.uuid4())
        filename = folder / f"{network}_{unique_id}.json"
        
        try:
            save_control_homophily(
                network,
                filename,
                n_control_iters=100,
                n_fitting_iters=10
            )
            print(f"Control analysis completed for {network}")
        except Exception as e:
            print(f"Error processing {network}: {e}")
            continue
    
    print("\nControl analysis completed for all networks")


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description='Network Homophily Analysis')
    parser.add_argument('--mode', choices=['example', 'control', 'both'], 
                       default='example', help='Analysis mode to run')
    parser.add_argument('--network', type=str, default='gene',
                       help='Network name for analysis')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for figures and results')
    
    args = parser.parse_args()
    
    # Set global output directory
    global OUTPUT_DIR
    OUTPUT_DIR = Path(args.output_dir)
    
    if args.mode in ['example', 'both']:
        run_analysis()
    
    if args.mode in ['control', 'both']:
        run_control_analysis()


if __name__ == "__main__":
    # If running without command line arguments, run example analysis
    if len(sys.argv) == 1:
        run_analysis()
    else:
        main()