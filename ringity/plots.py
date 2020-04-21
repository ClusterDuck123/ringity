import networkx as nx
import matplotlib.pyplot as plt

NODE_COL = [0/255, 85 /255, 100/255]
EDGE_COL = [0/255, 140/255, 160/255]

def ax_setup(ax):
    ax.tick_params(axis='both', which='major', labelsize=24)

    ax.spines['left'].set_linewidth(2.5)
    ax.spines['left'].set_color('k')

    ax.spines['bottom'].set_linewidth(2.5)
    ax.spines['bottom'].set_color('k')


def plot_nx(G,
            pos = None,
            ax  = None,
            node_colors = None,
            node_alpha  = 0.3,
            edge_colors = None,
            edge_alpha  = 0.2,
            silence = False,
            **kwargs):

    if pos is None:
        pos = nx.spring_layout(G)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8));
        fig.patch.set_alpha(0)
    if node_colors is None:
        node_colors = [NODE_COL]*nx.number_of_nodes(G)
    if edge_colors is None:
        edge_colors = [EDGE_COL]*nx.number_of_edges(G)
    nodes = nx.draw_networkx_nodes(G, pos=pos, alpha=node_alpha, ax=ax, node_color=node_colors, node_size=15, linewidths=1)
    edges = nx.draw_networkx_edges(G, pos=pos, alpha=edge_alpha, ax=ax, edge_color=edge_colors)
    ax.axis('off');

    if silence:
        plt.close()
