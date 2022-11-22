import numpy as np
import networkx as nx

from ringity.generators import point_clouds
from scipy.spatial.distance import pdist, squareform


def annulus(N, r, dist_th,
        noise = 0,
        seed = 0):
    """Construct geometric network from uniformly sampled annulus."""
    X = point_clouds.annulus(N = N, r = r, noise = noise, seed = seed)
    G = from_point_cloud(X, dist_th)
    return G


def from_point_cloud(X, dist_th):
    """Construct geometric network from point cloud."""
    D = squareform(pdist(X))
    A = np.where(np.abs(D) > dist_th, 0, 1)
    np.fill_diagonal(A,0)
    G = nx.from_numpy_array(A)
    return G