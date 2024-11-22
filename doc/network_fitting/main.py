#!/usr/bin/env python
# coding: utf-8

import os
import tqdm
import json
import scipy.stats
import numpy as np
import pandas as pd
import ringity as rg
import networkx as nx
import itertools as it
import matplotlib.pyplot as plt
import uuid

import sys
import os

import ringity
import ringity.networkfitting.retrieve_positions 

from ringity.networkfitting.retrieve_positions import PositionGraph
from ringity.networkfitting.fitting_model import FitNetwork

import json
import os
import sys

network_name = "lipid"#sys.argv[1]
network_model = "none"#sys.argv[2]
make_figures = True#bool(sys.argv[3])

def main(network_name,
         network_model,
         make_figures,
         folder):
    
    uuid_ = str(uuid.uuid4())
    
    folder = "test/" #f"{network_name}_{network_model}_{uuid}/"
    os.makedirs(folder, exist_ok=True)
    
    G_true = load_network(network_name)
    
    G,parameters = make_similar_network_model_random(G_true, network_model)
    
    positions,fitter = run_analysis(G)
    
    
    filename = "test.json" # f"{uuid}.json"
    save_values_to_json(folder, filename, parameters, fitter.c, fitter.w)
    
    if make_figures:
        
        if network_model=="none":
            color_dict = make_color_dict()
            color = color_dict[network_name]
        else:
            color = "k"
            
        save_figures(folder,G,fitter,positions,color=color)

def remove_selfloops(G):
    for u, v in G.edges():
        if u == v:
            G.remove_edge(u,v)

def load_network(name):
    G = nx.read_gml(f"data/empirical_networks/{name}.gml")
    remove_selfloops(G)
    G = nx.relabel_nodes(G, lambda x:str(x))
    return G

def get_largest_component_with_positions(G, positions):
    # Find the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    
    # Subset the graph to the largest connected component
    G_largest = G.subgraph(largest_cc).copy()
    
    # Get the node indices for the largest component
    largest_cc_nodes = list(largest_cc)
    
    # Get the positions corresponding to the nodes in the largest component
    largest_positions = [positions[node] for node in largest_cc_nodes]
    
    return G_largest, largest_positions


def make_similar_network_model_random(G_true, network_model):
    
    avg_degree = np.mean(list(dict(G_true.degree()).values()))
    
    N = len(G_true.nodes())
    density = 2*G_true.size()/(N*(N-1))
 
    if network_model == "none":
        G = G_true
        parameters={}
        

    if network_model == "erdos_renyi":
        p = density
        parameters = {"p":p}
        G = nx.erdos_renyi_graph(N,p)

            
    if network_model == "configuration":

        G = nx.configuration_model(list(dict(G_true.degree()).values()))
        parameters={}

    if network_model == "this_paper":

        beta=0.7+0.3*np.random.rand()
        r=0.5*np.random.rand()

        positions,G = rg.network_model(N,rho=density,beta=beta,a=r,return_positions=True)
        G, positions = get_largest_component_with_positions(G, positions)

        parameters = {"beta":beta,"r":r, "density":density}
            
    parameters["true_density"]=density
    parameters["true_avg_degree"]=avg_degree
    parameters["N"]=N

    return G,parameters

def run_analysis(G):
    
    self = PositionGraph(G)
    #self.positions = positions
    k = np.sqrt(2*np.pi/len(self.nodelist))
    self.make_circular_spring_embedding(k = k,verbose=True)
    self.smooth_neighborhood_widths()
    self.recenter_and_reorient()
    self.reconstruct_positions()


    fitter = FitNetwork(self.G,
                    self.rpositions
                )    
    fitter.links_by_distance()

    def slope_down(theta,c,w):
        temp = w-theta
        return c*0.5*(np.abs(temp)+temp)

    p,_ = scipy.optimize.curve_fit(slope_down,
                                   fitter.midpoints[1:],
                                   50*(fitter.counts_neighbors/fitter.counts_total)[1:],
                                   p0=None,
                                   sigma=None
                                  )

    fitter.c,fitter.w = p 
    fitter.c = fitter.c/50
    
    return self,fitter
    
    
    

    







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

    # Create the folder structure if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Create the JSON file path
    file_path = os.path.join(folder, filename)

    # Create a dictionary to store the values
    data = {
        "parameters":parameters,
        "fitter_c": fitter_c,
        "fitter_w": fitter_w
    }

    # Write the dictionary to the JSON file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def edge_points(ax, G, nodelist, p_dict,fontsize = 15,color='k'):

    xy = np.array([[p_dict[i], p_dict[j]] for i,j in G.edges()])
    swap = np.vstack([xy[:,1],xy[:,0]]).T
    xy = np.vstack([xy,swap])

    n_nodes = len(nodelist)
    ax.scatter(xy[:,0], xy[:,1], c=color,s=50/np.sqrt(len(G.edges())))

    ax.set_xlabel(r"position of node $i$",
                  fontsize=fontsize
                 )

    ax.set_ylabel(r"position of node $j$",
                  fontsize=fontsize
                 )

    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
                      fontsize=fontsize)
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_yticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
                      fontsize=fontsize)
    
    
def save_figures(folder,G,fitter,self,fontsize=20,color='k'):
    
    fig,ax = plt.subplots()
    edge_points(ax, self.G, self.nodelist, self.embedding_dict,fontsize=fontsize,color=color)
    ax.axis("tight")
    fig.savefig(folder+"fig_1_pos.png")

    fig,ax = plt.subplots()
    edge_points(ax, self.G, self.nodelist, self.rpositions,fontsize=fontsize,color=color)
    ax.axis("tight")
    fig.savefig(folder+"fig_2_pos.png")

    fig1 = fitter.links_by_distance()

    def slope_down(theta,c,w):
        temp = w-theta
        return c*0.5*(np.abs(temp)+temp)

    # [1:] because we want to avoid counting the absence of position_finder-loops
    p,_ = scipy.optimize.curve_fit(slope_down,
                                fitter.midpoints[1:],
                                50*(fitter.counts_neighbors/fitter.counts_total)[1:],
                                p0=None,
                                sigma=None
                                )

    fitter.c,fitter.w = p 
    fitter.c = fitter.c/50
    
    fig2 = fitter.neighbor_proportion()
    fig3 = fitter.draw_edge_edge_and_fit()

    fig1.savefig(folder+"fig_1_fit.png")
    fig2.savefig(folder+"fig_2_fit.png")
    fig3.savefig(folder+"fig_3_fit.png")

    fig,ax = plt.subplots()
    ax.scatter((fitter.counts_neighbors/fitter.counts_total)[1:],
                slope_down(fitter.midpoints[1:], fitter.c, fitter.w),c=color,s=300)
    ax.set_xlabel("True Proportion")
    ax.set_ylabel("Fit Function")
    ax.axis("tight")
    fig.savefig(folder+"/goodness_of_fit.png")

    fig,ax = plt.subplots()
    ax.hist(self.rpositions.values(),density=True,color=color,bins=int(np.sqrt(len(self.rpositions.values()))));
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
                    fontsize=fontsize)

    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_yticklabels([0, 0.1, 0.2, 0.3],
                    fontsize=fontsize)

    ax.set_xlabel("Position",fontsize=fontsize)
    ax.set_ylabel("Density",fontsize=fontsize)
    ax.axis("tight")
    fig.savefig(folder+"/rpos_bins.png")
    
    fig,ax = plt.subplots()
    ax.scatter((fitter.counts_neighbors/fitter.counts_total)[1:],
                slope_down(fitter.midpoints[1:], fitter.c, fitter.w),c=color,s=300)
    ax.set_xlabel("True Proportion")
    ax.set_ylabel("Fit Function")
    ax.axis("tight")
    fig.savefig(folder+"/goodness_of_fit.png")

    fig,ax = plt.subplots()
    ax.hist(self.rpositions.values(),density=True,color=color,bins=int(np.sqrt(len(self.rpositions.values()))));
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"],
                    fontsize=fontsize)

    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.set_yticklabels([0, 0.1, 0.2, 0.3],
                    fontsize=fontsize)

    ax.set_xlabel("Position",fontsize=fontsize)
    ax.set_ylabel("Density",fontsize=fontsize)

    ax.axis("tight")

    fig.savefig(folder+"fig_3_fit.png")
   
def make_color_dict():
    color_scheme = {"Jasmine":"#ffd07b",
                "Glaucous":"#577399",
                "Dark purple":"#412234",
                "Moss green":"#748e54",
                "Keppel":"#44bba4"
               }

    color_dict = {
    "immune":  color_scheme["Keppel"],
    "fibro":   color_scheme["Glaucous"],
    "gene":    color_scheme["Moss green"],
    "lipid":   color_scheme["Jasmine"],
    "soil":    color_scheme["Dark purple"],
    "gene_corrected":color_scheme["Moss green"]
    }
    
    return color_dict

   
import argparse


if __name__ == "__main__":
    
    

    parser = argparse.ArgumentParser(
                        prog='ProgramName',
                        description='What the program does',
                        epilog='Text at the bottom of help')
    

    parser.add_argument('--network')
    parser.add_argument('--model')
    parser.add_argument('--make_figs')
    parser.add_argument('--output_folder')
    

    args = parser.parse_args(sys.argv[1:])

    args.network = "lipid"
    args.model = "none"
    args.make_figs = "true" 
    
    print(args)
    
    network_name = args.network
    network_model = args.model
    make_figures = (args.make_figs == "true")
    folder=args.output_folder
    main(network_name,network_model,make_figures,folder)

