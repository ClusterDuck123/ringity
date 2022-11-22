from ringity.ringscore.ring_score_flavours import ring_score_from_sequence
from ringity.networkmeasures.centralities import net_flow, resistance
from ringity.classes.pdiagram import PersistenceDiagram
from ringity.readwrite.prompts import _yes_or_no, _assertion_statement
from ringity.classes.exceptions import UnknownGraphType
from gtda.homology import VietorisRipsPersistence

import os
import time
import subprocess
import scipy.sparse

import numpy as np
import networkx as nx

def ring_score(arg,
               argtype = 'pointcloud',
               flavour = 'geometric',
               base = None,
               nb_pers = np.inf,
               **kwargs):
    """Calculate ring score from various data structures.

    Generic function to calculate ring score by calling another appropriate
    ring score function:
        - If ``arg`` is a networkx graph, it will call ``ring_score_from_graph``.
        - If ``arg`` is a ringity persistence diagram, it will call 
        ``ring_score_from_persistence_diagram``.
        - If ``arg`` is a numpy array or sparse matrix the paramter ``argtype`` 
        specifies how to interpret the data and which ring-score function to call.
        
    All functions eventually call ``ring_score_from_sequence``.
    
    Parameters
    ----------
    arg : numpy array, sparse matrix, networkx graph or ringity persistence
    diagram.

    Returns
    -------
    score : float, ring-score of given object.
    """

    if isinstance(arg, nx.Graph):
        dgm = diagram(arg)
    elif isinstance(arg, PersistenceDiagram):
        dgm = arg
    elif isinstance(arg, np.ndarray) or isinstance(arg, scipy.sparse.spmatrix):
        if argtype == 'pointcloud':
            dgm = pdiagram_from_point_cloud(arg, **kwargs)
        else:
            raise Exception
    else:
        raise Exception

    score = ring_score_from_pdiagram(
                                    dgm = dgm,
                                    flavour = flavour,
                                    base = base,
                                    nb_pers = nb_pers)
    return score
                                                        
# -----------------------------------------------------------------------------    
# ---------------------------- RING SCORE FUNCTIONS ---------------------------
# -----------------------------------------------------------------------------     
def ring_score_from_point_cloud(X,
                                flavour = 'geometric',
                                base = None,
                                nb_pers = np.inf,
                                persistence = 'VietorisRipsPersistence',
                                metric='euclidean',
                                metric_params={},
                                homology_dim = 1,
                                **kwargs):
    """Calculate ring score from point cloud.

    Parameters
    ----------
    X : numpy array, sparse matrix

    Returns
    -------
    score : float, ring-score of given object.
    """
    dgm = pdiagram_from_point_cloud(X,
                                    persistence = persistence,
                                    metric = metric,
                                    metric_params = metric_params,
                                    homology_dim = homology_dim,
                                    **kwargs)
    return ring_score_from_pdiagram(dgm,
                                    flavour = flavour,
                                    base = base,
                                    nb_pers = nb_pers)
    
                                               
def ring_score_from_network(X,
                            flavour = 'geometric',
                            base = None,
                            nb_pers = np.inf,
                            persistence = 'VietorisRipsPersistence',
                            metric='euclidean',
                            metric_params={},
                            homology_dim = 1,
                            **kwargs):
    """Calculate ring score from network.

    Parameters
    ----------
    X : networkx graph

    Returns
    -------
    score : float, ring-score of given object.
    """
    dgm = pdiagram_from_point_cloud(X,
                                    persistence = persistence,
                                    metric = metric,
                                    metric_params = metric_params,
                                    homology_dim = homology_dim,
                                    **kwargs)
    return ring_score_from_pdiagram(dgm,
                                    flavour = flavour,
                                    base = base,
                                    nb_pers = nb_pers)
                                    
                                    
def ring_score_from_distance_matrix(dgm,
                                    flavour = 'geometric',
                                    nb_pers = np.inf,
                                    base = None):
    """Calculates ring-score from a PersistenceDiagram object."""
    pass
                            
                                    
def ring_score_from_pdiagram(dgm,
                             flavour = 'geometric',
                             nb_pers = np.inf,
                             base = None):
    """Calculates ring-score from a PersistenceDiagram object."""
    return ring_score_from_sequence(dgm.sequence,
                                    flavour = flavour,
                                    nb_pers = nb_pers,
                                    base = None)
                                               
# -----------------------------------------------------------------------------    
# ----------------------- PERSISTENCE DIAGRAM FUNCTIONS -----------------------
# ----------------------------------------------------------------------------- 
def pdiagram_from_point_cloud(X,
                              persistence = 'VietorisRipsPersistence',
                              metric='euclidean',
                              metric_params={},
                              homology_dim = 1,
                              **kwargs):
    """Constructs a PersistenceDiagram object from a point cloud.
    
    This function wraps persistent homolgy calculation from giotto-tda."""
    
    
    if persistence == 'VietorisRipsPersistence':
        VR = VietorisRipsPersistence(metric = metric,
                                     metric_params = metric_params,
                                     homology_dimensions = tuple(range(homology_dim+1)),
                                     **kwargs)
        dgm = VR.fit_transform([X])[0]
    return PersistenceDiagram.from_gtda(dgm, homology_dim = homology_dim)   
    
    
def pdiagram_from_graph(X, metric = 'net_flow', **kwargs):
    """Constructs a PersistenceDiagram object from a networkx graph.
    
    This function is not available yet."""
    pass           
    
    
def pdiagram_from_distance_matrix(dgm,):
    """Constructs a PersistenceDiagram object from a distance matrix.
    
    This function is not available yet."""
    pass         
    
# =============================================================================
#  -------------------------------- LEGACY ----------------------------------
# =============================================================================

# This code will be removed in future versions

def diagram(arg1 = None,
            verbose = False,
            metric = 'net_flow',
            distance_matrix = False,
            p = 1):
    """
    Returns the p-persistence diagram of a graph or a distance matrix. The graph
    is either given as a networkx graph or as a numpy array adjacency matrix. If
    a distance matrix is given, the flag distance_matrix=True has to be passed
    manually.
    The function is looking for an edge attribute specified by the keyword
    'metric'. If no edge attribute with this name is found, the function
    'get_distance_matrix' is invoked.
    """
    # warning_message = "The function ``diagram`` is depricated. Please use instead one " +
    #                   "of the diagram functions, like  ``persistence_diagram``,  "+
    #                   "``persistence_diagram_from_graph``, " +
    #                   "``persistence_diagram_from_point_cloud`` " +
    #                   "etc. "
    # warn(warning_message, DeprecationWarning, stacklevel = 2)

    if distance_matrix:
        D = arg1
    else:
        input_type = type(arg1)
        if input_type == np.ndarray:
            G = nx.from_numpy_array(arg1)
        elif input_type == nx.classes.graph.Graph:
            G = arg1
        else:
            raise UnknownGraphType(f"Unknown graph type {input_type}!")

        if nx.number_of_selfloops(G) > 0:
            if verbose:
                print('Self loops in graph detected. They will be removed!')
            G.remove_edges_from(nx.selfloop_edges(G))

        D = get_distance_matrix(G,
                                metric = metric,
                                verbose = verbose)

    t1 = time.time()
    VR = VietorisRipsPersistence(metric = 'precomputed',
                                 homology_dimensions = list(range(p+1)))
    dgm = VR.fit_transform([np.array(D)])[0]
    dgm = PersistenceDiagram.from_gtda(dgm)
    t2 = time.time()

    if verbose:
        print(f'Time for ripser calculation: {t2-t1:.6f} sec')

    return dgm


def get_distance_matrix(G, metric, verbose=False):
    """
    Returns numpy array of distances for each pair of nodes in the given graph.
    Distances are specified by the keyword 'metric'.
    """
    if   metric == 'resistance':
        return resistance(G)
    elif metric == 'SPL':
        return nx.floyd_warshall_numpy(G, weight=None)
    else:
        if not nx.get_edge_attributes(G, metric):
            induce_weight(G=G,
                          weight=metric,
                          verbose=verbose)

        t1 = time.time()
        D  = nx.floyd_warshall_numpy(G, weight=metric)
        t2 = time.time()

        if verbose:
            print(f'Time for SPL calculation: {t2-t1:.6f} sec')

        return D


def existing_edge_attribute_warning(weight):
    answer = input(
            f"Graph has already an edge attribute called '{weight}'! "
            f"Do you want to overwrite this edge attribute? ")
    proceed = _yes_or_no(answer)
    if not proceed:
        print('Process terminated!')
        return 'TERMINATED'
    else:
        return 0


def induce_weight(G, weight = 'net_flow', verbose=False):
    if verbose and nx.get_edge_attributes(G, weight):
        exit_status = existing_edge_attribute_warning(weight)
        if exit_status:
            return exit_status

    if   weight == 'net_flow':
        bb = net_flow(G)
    elif weight == 'betweenness':
        bb = nx.edge_betweenness_centrality(G)
    elif weight == 'current_flow':
        bb = nx.edge_current_flow_betweenness_centrality(G)
    else:
        raise Exception(f"Weight '{weight}' unknown!")

    nx.set_edge_attributes(G, values=bb, name=weight)
