import scipy
import numpy as np
import networkx as nx
import ringity.networks.centralities as cents

from scipy.sparse.csgraph import floyd_warshall
from scipy.spatial.distance import pdist, squareform, is_valid_dm

METRIC_TRANSFORMERS = {'resistance', 'spl'}

def pwdistance(data, 
            data_structure = 'suggest', 
            metric = None,
            verbose = False,):
    """Calculates pairwise distances from given data structure.

    Parameters
    ----------
    data : networkx.Graph, nump.ndarray
        Data to calculate pairwise-distance from.
    data_structure : str, optional
        Specify what kind of data the argument ``data`` describes.
        By default 'suggest'.
    metric : string, optional
        Which metric to use for pairwise distance caluclation. By default None.
    verbose : bool, optional
        Prints logs of various calulation steps. By default False.

    Raises
    ------
    Exception
        When data type of ``data`` is unknown.
    Exception
        When the string ``data_structure`` is unknown.
    """    
    
    if isinstance(data, nx.Graph):
        D = pwdistance_from_network(data, metric = metric)
    if isinstance(data, np.ndarray) or isinstance(data, scipy.sparse.spmatrix):
        if data_structure == 'suggest':
            if is_valid_dm(data):
                data_structure = 'adjacency_matrix'
            else:
                data_structure = 'point_cloud'
            if verbose:
                print(f"`data_structure` was set to {data_structure}")
                
        if data_structure == 'point_cloud':
            D = pwdistance_from_point_cloud(metric, verbose)
        elif data_structure == 'adjacency_matrix':
            D = pwdistance_from_adjacency_matrix(data)
        else:
            raise Exception(f"Data structure `{data_structure} unknown.")
    else:
        raise Exception(f"Data type `{type(data)} unknown.")

def pwdistance_from_point_cloud(X, 
                            metric = 'euclidean', 
                            verbose = False,
                            **kwargs):
    if metric is None:
        metric = 'euclidean'
    D = squareform(pdist(X, metric = metric))
    return D

def pwdistance_from_adjacency_matrix(A, 
                                verbose = False):
    """Uses Floyd-Warshall algorithm to calculate all shortest path length matrix 
    from a given adjacency matrix.
    """
    is_valid = is_valid_dm(A)
    if not is_valid:
        raise ValueError('Adjacency matrix `A` must be symmetric and have zeros on the diagonal.')
    D = floyd_warshall(A)
    return D


def pwdistance_from_network(
    G,
    metric="net_flow",
    use_weights=None,
    store_weights=True,
    remove_self_loops=True,
    verbose=False,
):
    """Calculate pairwise distances from a given network.

    Parameters
    ----------
    G : _type_
        _description_
    metric : str, optional
        _description_, by default 'net_flow'
    use_weights : _type_, optional
        _description_, by default None
    store_weights : bool, optional
        Stores calculated weights as edge attributes if `metric`
        corresponds to a centrality measure. By default True.
    new_weight_name : _type_, optional
        _description_, by default None
    overwrite_weights : bool, optional
        _description_, by default False
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    if not store_weights:
        G = G.copy()

    if (nx.number_of_selfloops(G) > 0) and remove_self_loops:
        if verbose:
            print("Self loops in graph detected. They will be removed!")
        G.remove_edges_from(nx.selfloop_edges(G))
    if isinstance(use_weights, str):
        metric = use_weights
        if verbose:
            print('metric was set to use_weights.')

    # Check if metric is stored as edge attribute
    induction_status = _check_weight_induction(
        G, metric=metric, use_weights=use_weights, verbose=verbose
    )
    if induction_status:
        induce_edge_weights(G, metric=metric, verbose=verbose)

    spl_status = _check_spl_calculation(
        G, induction_status=induction_status, metric=metric, verbose=verbose
    )
    if spl_status:
        A = nx.to_numpy_array(G, weight=metric)
        D = floyd_warshall(A)
    else:
        if metric.lower() == "resistance":
            D = cents.resistance(G)
        elif metric.lower() == "spl":
            A = nx.to_numpy_array(G, weight=None)
            D = floyd_warshall(A)
    return D


def induce_edge_weights(G, metric = 'net_flow', verbose = False):
    if metric.lower() == 'net_flow':
        ew_dict = cents.net_flow(G)
    elif metric.lower() == 'betweenness':
        ew_dict = nx.edge_betweenness_centrality(G)
    elif metric.lower() == 'current_flow':
        ew_dict = cents.current_flow(G)
    else:
        raise Exception(f'Centrality measure {metric} unknown.')
    nx.set_edge_attributes(G, ew_dict, metric)


def _check_weight_induction(G, metric, use_weights, verbose):
    # Cases where no further calculation is needed.
    if use_weights is False:
        if verbose:
            print(
                f'No weights will be used for calculations.')
        return False
    elif nx.get_edge_attributes(G, metric):
        if verbose and use_weights:
            print(
                f'Weights named `{metric}` detected. ' 
                f'They will be used for distance calculation.')
        return False
    elif use_weights is True:
        raise Exception(f'No weights named {metric} detected.')
    
    # Cases where calculation is needed depending on the metric.
    elif metric.lower() in METRIC_TRANSFORMERS:
        if verbose:
            print(
            f'No weights named `{metric}` detected. ' 
            f'Metric `{metric}` will be directly calculated.')
        return False
    else:
        # Only case where weights are being induced.
        if verbose and metric.lower() not in METRIC_TRANSFORMERS:
            print(
                f'No weights named `{metric}` detected. ' 
                f'Centrality measure `{metric}` will be calculated.')
        return True

def _check_spl_calculation(G, induction_status, metric, verbose):
    if induction_status:
        if verbose:
            print(
                f'Edge weight `{metric}` was successfully induced. '
                f'Weights `{metric}` will be extended to metric space using SPLs.')
        return True
    elif metric not in METRIC_TRANSFORMERS:
        if verbose:
            print(f'Weights `{metric}` will be extended to metric space using SPLs.')
        return True
    else:
        if verbose:
            print(f'Metric `{metric}` will be used for pairwise distance calculations.')
        return False
