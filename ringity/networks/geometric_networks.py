from ringity.utils import point_clouds

import numpy as np
import networkx as nx
import scipy.sparse.coo as coo

from functools import partial
from scipy.spatial.distance import pdist, squareform

def circle(N,
        noise = 0,
        er_fc_th = 10,
        rel_dist_th = None,
        abs_dist_th = None,
        density_th = None,
        n_neighbors = None, 
        seed = None,
        return_point_cloud = False,
        keep_weights = False,
        new_weight_name = 'distance',
        verbose = False):
    """Construct geometric network from uniformly sampled circle."""
    X = point_clouds.circle(N = N, noise = noise, seed = seed)
    G = pointcloud_to_graph(X, 
                er_fc_th = er_fc_th,
                rel_dist_th = rel_dist_th,
                abs_dist_th = abs_dist_th,
                density_th = density_th,
                n_neighbors = n_neighbors, 
                keep_weights = keep_weights,
                new_weight_name = new_weight_name,
                verbose = verbose)
    if return_point_cloud:
        return G, X
    else:
        return G

def vonmises_circles(N, kappa, 
        noise = 0,
        er_fc_th = 10,
        rel_dist_th = None,
        abs_dist_th = None,
        density_th = None,
        n_neighbors = None, 
        seed = None,
        return_point_cloud = False,
        keep_weights = False,
        new_weight_name = 'distance',
        verbose = False):
    """Construct geometric network from von Mises distributed 
    the of circle."""
    X = point_clouds.vonmises_circles(N = N, kappa = kappa, noise = noise, seed = seed)
    G = pointcloud_to_graph(X, 
                er_fc_th = er_fc_th,
                rel_dist_th = rel_dist_th,
                abs_dist_th = abs_dist_th,
                density_th = density_th,
                n_neighbors = None, 
                keep_weights = keep_weights,
                new_weight_name = new_weight_name)
    if return_point_cloud:
        return G, X
    else:
        return G


def annulus(N, r, 
        noise = 0,
        er_fc_th = 10,
        rel_dist_th = None,
        abs_dist_th = None,
        density_th = None,
        n_neighbors = None, 
        seed = None,
        return_point_cloud = False,
        keep_weights = False,
        new_weight_name = 'distance',
        verbose = False):
    """Construct geometric network from unifomly sampled annulus."""
    X = point_clouds.annulus(N = N, r = r, noise = noise, seed = seed)
    G = pointcloud_to_graph(X, 
                er_fc_th = er_fc_th,
                rel_dist_th = rel_dist_th,
                abs_dist_th = abs_dist_th,
                density_th = density_th,
                n_neighbors = None, 
                keep_weights = keep_weights,
                new_weight_name = new_weight_name)
    if return_point_cloud:
        return G, X
    else:
        return G

def cylinder(N, height,
        noise = 0,
        er_fc_th = 10,
        rel_dist_th = None,
        abs_dist_th = None,
        density_th = None,
        n_neighbors = None, 
        seed = None,
        return_point_cloud = False,
        keep_weights = False,
        new_weight_name = 'distance',
        verbose = False):
    """Construct geometric network from unifomly sampled cylinder."""
    X = point_clouds.cylinder(N = N, height = height, noise = noise, seed = seed)
    G = pointcloud_to_graph(X, 
                er_fc_th = er_fc_th,
                rel_dist_th = rel_dist_th,
                abs_dist_th = abs_dist_th,
                density_th = density_th,
                n_neighbors = n_neighbors,
                keep_weights = keep_weights,
                new_weight_name = new_weight_name,
                verbose = verbose)
    if return_point_cloud:
        return G, X
    else:
        return G

def torus(N, r,
        noise = 0,
        er_fc_th = 10,
        rel_dist_th = None,
        abs_dist_th = None,
        density_th = None,
        n_neighbors = None, 
        seed = None,
        return_point_cloud = False,
        keep_weights = False,
        new_weight_name = 'distance',
        verbose = False):
    """Construct geometric network from unifomly sampled torus."""
    X = point_clouds.torus(N = N, r = r, noise = noise, seed = seed)
    G = pointcloud_to_graph(X, 
                er_fc_th = er_fc_th,
                rel_dist_th = rel_dist_th,
                abs_dist_th = abs_dist_th,
                density_th = density_th,
                n_neighbors = n_neighbors, 
                keep_weights = keep_weights,
                new_weight_name = new_weight_name,
                verbose = verbose,
                )
    if return_point_cloud:
        return G, X
    else:
        return G


def pointcloud_to_graph(X,
                er_fc_th = 10,
                rel_dist_th = None,
                abs_dist_th = None,
                density_th = None,
                n_neighbors = None, 
                keep_weights = False,
                new_weight_name = 'distance',
                verbose = False,
                ):
    """Construct geometric network from point cloud."""

    Dsq = pdist(X)
    D = squareform(Dsq)

    if n_neighbors:
        if verbose:
            print("Calling distance_to_knn_graph().")
        G = distance_to_knn_graph(D,
                n_neighbors = n_neighbors,
                keep_weights = keep_weights,
                new_weight_name = new_weight_name,
                verbose = verbose)
    else:
        if verbose:
            print("Calling distance_to_sublevel_graph().")
        G = distance_to_sublevel_graph(D,
                er_fc_th = er_fc_th,
                rel_dist_th = rel_dist_th,
                abs_dist_th = abs_dist_th,
                density_th = density_th, 
                keep_weights = keep_weights,
                new_weight_name = new_weight_name,
                verbose = verbose)
    
    return G

def distance_to_knn_graph(D, 
                n_neighbors = None,
                keep_weights = False,
                new_weight_name = 'distance',
                verbose = False):
    d_th = _get_k_smallest_value_by_col(D, n_neighbors+1)
    mask = np.where(D > d_th, 0, 1)

    if keep_weights:
        A = np.where(mask + mask.T > 0, D, 0)
        G = nx.from_scipy_sparse_matrix(A, edge_attribute = new_weight_name)
    else:
        A = np.where(mask + mask.T > 0, 1, 0)
        np.fill_diagonal(A, 0)
        G = nx.from_numpy_array(A + A.T)
        for (u, v, d) in G.edges(data = True):
            d.clear()
    return G

def distance_to_sublevel_graph(D,
                er_fc_th = 10,
                rel_dist_th = None,
                abs_dist_th = None,
                density_th = None, 
                keep_weights = False,
                new_weight_name = 'distance',
                verbose = False):
    d_th = _get_threshold(squareform(D), 
                    er_fc_th = er_fc_th, 
                    rel_dist_th = rel_dist_th, 
                    abs_dist_th = abs_dist_th,
                    density_th = density_th)

    if keep_weights:
        A = coo.coo_matrix(np.where(D > d_th, 0, D))
        G = nx.from_scipy_sparse_matrix(A, edge_attribute = new_weight_name)
    else:
        A = np.where(D > d_th, 0, 1)
        np.fill_diagonal(A, 0)
        G = nx.from_numpy_array(A)
        for (u, v, d) in G.edges(data = True):
            d.clear()
    return G

# ---------------------------------------------------------------------------------
# ------------------------------ AUXILIARY FUNCTIONS ------------------------------
# ---------------------------------------------------------------------------------

def _get_k_smallest_value_in_vec(vec, k):
    return np.partition(vec, k-1)[k-1]

def _get_k_smallest_value_by_col(D, k):
    k_smallest_values = np.apply_along_axis(partial(_get_k_smallest_value_in_vec, k = k), 
                                axis = 0, 
                                arr = D)
    return k_smallest_values

def _get_threshold(Dsq, rel_dist_th, abs_dist_th, density_th, er_fc_th):
    non_fc_th_types = [
                ('relative_distance_th',  rel_dist_th), 
                ('absolute_distance_th', abs_dist_th), 
                ('density_th', density_th),
                ]
    th_types = [(k,v) for k,v in non_fc_th_types if v is not None]

    if len(th_types) > 1:
        raise Exception(f'Conflicting thresholds given: {dict(th_types)}')
    
    elif len(th_types) == 1:
        (th_type, th), = th_types
        
    else:
        assert (er_fc_th != None), 'Please provide a threshold!'
        th_type = 'er_foldchange_th'
        th = er_fc_th

    if th_type == 'absolute_distance_th':
        d_th = th
    elif th_type == 'relative_distance_th':
        d_th = Dsq.max() * th
    elif th_type == 'density_th':
        d_th = np.quantile(Dsq, th)
    elif th_type == 'er_foldchange_th':
        N = round((1 + np.sqrt(1 + 8*len(Dsq))) / 2)
        density = th * np.log(N)/N
        d_th = np.quantile(Dsq, density)
    elif th_type == 'n_neighbors':
        N = round((1 + np.sqrt(1 + 8*len(Dsq))) / 2)
        density = th * np.log(N)/N
        d_th = np.quantile(Dsq, density)
    else:
        raise Exception('Something went wrong.')
    return d_th