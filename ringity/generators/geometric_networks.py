import numpy as np
import networkx as nx
import scipy.sparse.coo as coo

from ringity.generators import point_clouds
from scipy.spatial.distance import pdist, squareform


def circle(N,
        abs_th = None,
        rel_th = 0.25,
        noise = 0,
        return_point_cloud = False,
        seed = None):
    """Construct geometric network from uniformly sampled annulus."""
    X = point_clouds.circle(N = N, noise = noise, seed = seed)

    d_th = _get_threshold(rel_th, abs_th, X)
    G = from_point_cloud(X, d_th)
    
    if return_point_cloud:
        return G, X
    else:
        return G


def annulus(N, r, 
        abs_th = None,
        rel_th = 0.25,
        noise = 0,
        return_point_cloud = False,
        seed = None):
    """Construct geometric network from uniformly sampled annulus."""
    X = point_clouds.annulus(N = N, r = r, noise = noise, seed = seed)

    d_th = _get_threshold(rel_th, abs_th, X)
    G = from_point_cloud(X, d_th)
    
    if return_point_cloud:
        return G, X
    else:
        return G

def cylinder(N, height,
        abs_th = None,
        rel_th = 0.25,
        noise = 0,
        return_point_cloud = False,
        seed = None):
    """Construct geometric network from uniformly sampled cylinder."""
    X = point_clouds.cylinder(N = N, height = height, noise = noise, seed = seed)

    d_th = _get_threshold(rel_th, abs_th, X)
    G = from_point_cloud(X, d_th)
    
    if return_point_cloud:
        return G, X
    else:
        return G


def from_point_cloud(X, dist_th, 
                keep_weights = False,
                new_weight_name = 'distance'):
    """Construct geometric network from point cloud."""

    D = squareform(pdist(X))
    if keep_weights:
        A = coo.coo_matrix(np.where(D > dist_th, 0, D))
        G = nx.from_scipy_sparse_matrix(A, edge_attribute = new_weight_name)
    else:
        A = np.where(D > dist_th, 0, 1)
        np.fill_diagonal(A, 0)
        G = nx.from_numpy_array(A)
        for (a, b, d) in G.edges(data = True):
            d.clear()
    return G

def _get_threshold(rel_th, abs_th, X):
    if abs_th is not None:
        d_th = abs_th
    else:
        d_th = pdist(X).max() * rel_th
    return d_th