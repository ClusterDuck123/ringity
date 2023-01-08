import numpy as np
import networkx as nx
import scipy.sparse.coo as coo

from ringity.generators import point_clouds
from scipy.spatial.distance import pdist, squareform


def circle(N,
        noise = 0,
        er_fc_th = 10,
        rel_dist_th = None,
        abs_dist_th = None,
        density_th = None,
        seed = None,
        return_point_cloud = False,
        keep_weights = False,
        new_weight_name = 'distance'):
    """Construct geometric network from uniformly sampled circle."""
    X = point_clouds.circle(N = N, noise = noise, seed = seed)
    G = from_point_cloud(X, 
                er_fc_th = er_fc_th,
                rel_dist_th = rel_dist_th,
                abs_dist_th = abs_dist_th,
                density_th = density_th,
                keep_weights = keep_weights,
                new_weight_name = new_weight_name)
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
        seed = None,
        return_point_cloud = False,
        keep_weights = False,
        new_weight_name = 'distance'):
    """Construct geometric network from von Mises distributed 
    the of circle."""
    X = point_clouds.vonmises_circles(N = N, kappa = kappa, noise = noise, seed = seed)
    G = from_point_cloud(X, 
                er_fc_th = er_fc_th,
                rel_dist_th = rel_dist_th,
                abs_dist_th = abs_dist_th,
                density_th = density_th,
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
        seed = None,
        return_point_cloud = False,
        keep_weights = False,
        new_weight_name = 'distance'):
    """Construct geometric network from unifomly sampled annulus."""
    X = point_clouds.annulus(N = N, r = r, noise = noise, seed = seed)
    G = from_point_cloud(X, 
                er_fc_th = er_fc_th,
                rel_dist_th = rel_dist_th,
                abs_dist_th = abs_dist_th,
                density_th = density_th,
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
        seed = None,
        return_point_cloud = False,
        keep_weights = False,
        new_weight_name = 'distance'):
    """Construct geometric network from unifomly sampled cylinder."""
    X = point_clouds.cylinder(N = N, height = height, noise = noise, seed = seed)
    G = from_point_cloud(X, 
                er_fc_th = er_fc_th,
                rel_dist_th = rel_dist_th,
                abs_dist_th = abs_dist_th,
                density_th = density_th,
                keep_weights = keep_weights,
                new_weight_name = new_weight_name)
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
        seed = None,
        return_point_cloud = False,
        keep_weights = False,
        new_weight_name = 'distance'):
    """Construct geometric network from unifomly sampled torus."""
    X = point_clouds.torus(N = N, r = r, noise = noise, seed = seed)
    G = from_point_cloud(X, 
                er_fc_th = er_fc_th,
                rel_dist_th = rel_dist_th,
                abs_dist_th = abs_dist_th,
                density_th = density_th,
                keep_weights = keep_weights,
                new_weight_name = new_weight_name)
    if return_point_cloud:
        return G, X
    else:
        return G


def from_point_cloud(X,
                er_fc_th = 10,
                rel_dist_th = None,
                abs_dist_th = None,
                density_th = None,
                keep_weights = False,
                new_weight_name = 'distance'):
    """Construct geometric network from point cloud."""

    Dsq = pdist(X)
    D = squareform(Dsq)

    d_th = _get_threshold(Dsq, 
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
        for (a, b, d) in G.edges(data = True):
            d.clear()
    return G

def _get_threshold(Dsq, rel_dist_th, abs_dist_th, density_th, er_fc_th):
    non_fc_th_types = [
                ('relative_distance_th',  rel_dist_th), 
                ('absolute_distance_th', abs_dist_th), 
                ('density_th', density_th)]
    th_types = [(k, v) for k,v in non_fc_th_types if v is not None]

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
    else:
        raise Exception('Something went wrong.')
    return d_th