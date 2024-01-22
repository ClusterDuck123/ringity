import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from collections import Counter
from ringity.utils.plotting.styling import ax_setup
from ringity.utils.plotting._plotly import plot_nx_plotly_3d, plot_nx_plotly_2d

CEMM_COL1 = (  0/255,  85/255, 100/255)
CEMM_COL2 = (  0/255, 140/255, 160/255)
CEMM_COL3 = ( 64/255, 185/255, 212/255)
CEMM_COL4 = (212/255, 236/255, 242/255)

DARK_CEMM_COL1 = (0/255, 43/255, 50/255)
BAR_COL = (0.639, 0.639, 0.639)


def plot(arg, ax = None, **kwargs):
    if isinstance(arg, np.ndarray):
        plot_X(arg, ax = ax, **kwargs)
    if isinstance(arg, nx.Graph):
        plot_nx(arg, ax = ax, **kwargs)

def plot_seq(dgm, trim = None, ax = None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
        fig.patch.set_alpha(0)

    if trim is None:
        dgm_plot = dgm.copy()
    else:
        dgm_plot = dgm.trimmed(trim)

    ax_setup(ax)
    bar = list(dgm_plot.sequence)
    ax.bar(range(len(bar)), bar, color = BAR_COL);

def plot_nx(G,
            pos = None,
            ax  = None,
            library = None,
            dim = 2,
            hoverinfo = None,
            node_color = None,
            node_colors = None,
            node_alpha  = 0.3,
            edge_colors = None,
            edge_alpha  = 0.2,
            **kwargs):
    
    if library is None:
        if dim == 2:
            library = 'matplotlib'
        elif dim == 3:
            library = 'plotly'
    
    if library == 'matplotlib':
        return plot_nx_old(G,
            pos = pos,
            ax  = ax,
            node_colors = node_colors,
            node_alpha  = node_alpha,
            edge_colors = edge_colors,
            edge_alpha  = edge_alpha,
            **kwargs)
    if library == 'plotly':
        if dim == 2:
            return plot_nx_plotly_2d(G, 
                    pos = pos, 
                    hoverinfo = hoverinfo, 
                    node_color = node_color)
        elif dim == 3:
            return plot_nx_plotly_3d(G, 
                    pos = pos, 
                    hoverinfo = 
                    hoverinfo, 
                    node_color = node_color)

def plot_nx_old(G,
            pos = None,
            ax  = None,
            node_colors = None,
            node_alpha  = 0.3,
            edge_colors = None,
            edge_alpha  = 0.2,
            **kwargs):
    """Plots a networkx graph.

    Parameters
    ----------
    G : NetworkX Graph

    pos : dictionary, optional
        A dictionary with nodes as keys and positions as values.
        If not specified a spring layout positioning will be computed.
    ax : Matplotlib Axes object, optional
        Draw the graph in specified Matplotlib axes.
    node_colors : RGB tuple, optional
        Color of nodes, by default ???.
    node_alpha : float, optional
        Transparency parameter for nodes, by default 0.3.
    edge_colors : RGB tuple, optional
        Color of edges, by default ???.
    edge_alpha : float, optional
        Transparency parameter for nodes, by default 0.2
    """      

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


def plot_dgm(dgm, 
        ax = None, 
        return_artist_obj = False,
        **kwargs):
    """Plots persistence diagram. 

    Parameters
    ----------
    dgm : _type_
        _description_
    ax : Matplotlib Axes object, optional
        Draw the graph in specified Matplotlib axes.
        In ``None``, a new matplotlib figure will be created and 
        ``**kwargs`` will be passed on to ``plt.subplots()``.
    return_artist_obj : bool, optional
        If `return_artist_obj` is `True`, returns all artist objects that 
        have been created of modified. If a figure is created from scratch,
        it will return a pair of ``(fig, ax)``, otherwise it will only return 
        `ax`.

    Returns
    -------
    See parameter ``return_artist_obj``.
    """
    
    x,y = zip(*[(k.birth,k.death) for k in dgm])
    d = max(y)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
        fig.patch.set_alpha(0)

    ax_setup(ax)

    hw = 0.025 # head width of the arrow

    ax.set_xlim([-hw, d*1.1])
    ax.set_ylim([-hw, d*1.1])

    ax.plot(x, y, '*', markersize = 5, color = CEMM_COL2)
    ax.plot([0,d],[0,d], color = DARK_CEMM_COL1,
                         linewidth = 1,
                         linestyle = 'dashed')

    return_data = _parse_return_data(return_artist_obj, fig, ax)
    return return_data


def plot_X(X, ax = None, return_artist_obj = False, **kwargs):
    n_obs, n_vars = X.shape
    
    fig = None
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
        fig.patch.set_alpha(0)
    
    if n_vars == 2:
        x, y = X.T
        ax.scatter(x, y)
    
    ax_setup(ax)
    ax.axis('off')

    return_data = _parse_return_data(return_artist_obj, fig, ax)
    return return_data


def plot_degree_distribution(G, 
                        ax = None, 
                        figsize = None,
                        return_artist_obj = False):

    degree_sequence = sorted([d for n, d in G.degree()])
    degreeCount = Counter(degree_sequence)
    degs, cnts = zip(*degreeCount.items())

    fig = None
    if ax is None:
        if figsize is None:
            figsize = (8,6)
        fig, ax = plt.subplots(figsize = figsize)
        fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    ax.bar(degs, cnts, 
        width = 1, 
        color = CEMM_COL2, 
        linewidth = 0.75, 
        edgecolor = CEMM_COL4);

    ax_setup(ax)

    return_data = _parse_return_data(return_artist_obj, fig, ax)
    return return_data


def _parse_return_data(return_artist_obj, fig, ax):
    if not return_artist_obj:
        return None
    
    return_data = tuple(obj for obj in (fig, ax) if obj is not None)

    return return_data



# TO BE IMPLEMENTED
def my_plotter(ax, data1, data2, param_dict):
    """
    A helper function to make a graph.
    """
    out = ax.plot(data1, data2, **param_dict)
    return out