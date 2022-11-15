import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from ringity.plotting.styling import ax_setup

CEMM_COL1 = (  0/255,  85/255, 100/255)
CEMM_COL2 = (  0/255, 140/255, 160/255)
CEMM_COL3 = ( 64/255, 185/255, 212/255)
CEMM_COL4 = (212/255, 236/255, 242/255)

DARK_CEMM_COL1 = (0/255, 43/255, 50/255)
BAR_COL = (0.639, 0.639, 0.639)


def plot(arg, ax = None):
    if isinstance(arg, np.ndarray):
        plot_X(arg, ax=ax)

def plot_seq(dgm, crop = None, ax = None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)

    if crop is None:
        dgm_plot = dgm.copy()
    else:
        dgm_plot = dgm.crop(crop)

    ax_setup(ax)
    bar = list(dgm_plot.sequence)
    ax.bar(range(len(bar)), bar, color = BAR_COL);


def plot_nx(G,
            pos = None,
            ax  = None,
            node_colors = None,
            node_alpha  = 0.3,
            edge_colors = None,
            edge_alpha  = 0.2,
            **kwargs):

    if pos is None:
        pos = nx.spring_layout(G)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8));
        fig.patch.set_alpha(0)
    if node_colors is None:
        node_colors = [CEMM_COL1]*nx.number_of_nodes(G)
    if edge_colors is None:
        edge_colors = [CEMM_COL2]*nx.number_of_edges(G)
    nodes = nx.draw_networkx_nodes(G, pos=pos,
                                      alpha=node_alpha,
                                      ax=ax,
                                      node_color=node_colors,
                                      node_size=15,
                                      linewidths=1)

    edges = nx.draw_networkx_edges(G, pos=pos,
                                      alpha=edge_alpha,
                                      ax=ax,
                                      edge_color=edge_colors)
    ax.axis('off');


def plot_dgm(dgm, ax=None, **kwargs):
    x,y = zip(*[(k.birth,k.death) for k in dgm])
    d = max(y)

    if ax is None:
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)

    ax_setup(ax)

    hw = 0.025 # head width of the arrow

    ax.set_xlim([-hw, d*1.1])
    ax.set_ylim([-hw, d*1.1])

    ax.plot(x, y, '*', markersize = 5, color = CEMM_COL2);
    ax.plot([0,d],[0,d], color=DARK_CEMM_COL1,
                         linewidth=1,
                         linestyle='dashed');


def plot_X(X, ax = None):
    n_obs, n_vars = X.shape
    
    if ax is None:
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)
    
    if n_vars == 2:
        x, y = X.T
        plt.scatter(x, y)
    
    ax_setup(ax)
    ax.axis('off');
