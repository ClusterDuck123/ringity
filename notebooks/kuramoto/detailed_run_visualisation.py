from main import MyModInstance,Run
import ringity 
import kuramoto
import matplotlib.pyplot as plt
import uuid
import os
import pandas as pd
import tqdm
import networkx as nx 
import numpy as np
from celluloid import Camera
import imageio  # To save as GIF

def main(input_folder):
    
    input_folder
    
    

def draw_phase_coherence(run):
    #activity = run["activity"]
    
    T  = run.T
    dt = run.dt
    
    timeframe = np.linspace(0,T,int(T//dt)+1)

    #pd.read_csv(f"data/intermediate/searching_for_incoherence/{folder}/runs/{run_folder}/full.csv",index_col=0)
    
    fig,ax = plt.subplots()
    ax.plot(timeframe,
            run.phase_coherence.flatten(),
            c="k"
            )
    
    ax.set_xlabel("time",fontsize=20)
    ax.set_ylabel(r"R",fontsize=20)
    ax.set_ylim((0,1))
    

    fig.suptitle(run.phase_coherence.flatten()[-10000:].std())
    
    
    return fig

def draw_activity_on_net(network, run, timestep, output_file):
    
    G = nx.from_numpy_array(network.adj_mat)

    activity = run.activity
    T = run.T
    dt = run.dt
    
    timeframe = np.linspace(0,T,int(T//dt)+1)
    n_timesteps = int(T//dt)+1
    
    t = timeframe[timestep]

    pos = {i:(np.cos(v),np.sin(v))
           for i,v in enumerate(network.positions.flatten())}
    
    G = nx.from_numpy_array(network.adj_mat)
    
    m = nx.draw_networkx_nodes(G,
                               pos=pos,
                               node_color = np.mod(activity[:,timestep],2*np.pi),
                               node_size=100,
                               cmap="Spectral"
                               )
    
    nx.draw_networkx_edges(G,pos=pos)
    plt.colorbar(m,label=f"Node Phase at time {t}")
    plt.savefig(output_file)

def draw_gif(output_file, network, natfreqs, activity,stride=100):
    #
    G = nx.from_numpy_array(network.adj_mat)
    
    # Create a figure and camera
    fig, ax = plt.subplots()
    camera = Camera(fig)

    for timestep in tqdm.tqdm(range(0, activity.shape[1], stride), total=(activity.shape[1] // stride)):
        
        pos = {i:(np.cos(v),np.sin(v)) for i,v in enumerate(np.mod(activity[:,timestep],2*np.pi))}
        m = nx.draw_networkx_nodes(G,pos=pos,node_color = natfreqs, node_size=100,cmap="Spectral")
        nx.draw_networkx_edges(G,pos=pos)
        fig.suptitle(timestep)
        camera.snap()

    plt.colorbar(m,label="Natural Frequencies")
    # Create animation
    animation = camera.animate()



    # Save as GIF
    animation.save(output_file, writer="pillow")
    print("GIF is written")
    
def draw_gif_positions(output_file, network, positions, activity,stride=100):
    #
    G = nx.from_numpy_array(network.adj_mat)
    
    # Create a figure and camera
    fig, ax = plt.subplots()
    camera = Camera(fig)

    for timestep in tqdm.tqdm(range(0, activity.shape[1], stride), total=(activity.shape[1] // stride)):
        
        pos = {i:(np.cos(v),np.sin(v)) for i,v in enumerate(np.mod(activity[:,timestep],2*np.pi))}
        m = nx.draw_networkx_nodes(G,pos=pos,node_color = positions, node_size=100,cmap="Spectral")
        nx.draw_networkx_edges(G,pos=pos)
        fig.suptitle(timestep)
        camera.snap()

    plt.colorbar(m,label="Original Position")
    # Create animation
    animation = camera.animate()


    # Save as GIF
    animation.save(output_file, writer="pillow")
    print("GIF is written")
    
main()