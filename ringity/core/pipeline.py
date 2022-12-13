from ringity.classes.pdiagram import PDiagram
from ringity.core.metric2ringscore import ring_score_from_sequence
from ringity.core.data2metric import pwdistance_from_network

from gtda.homology import VietorisRipsPersistence
from scipy.spatial.distance import is_valid_dm

import numpy as np
import networkx as nx
import scipy.sparse

# -----------------------------------------------------------------------------    
# -------------------- GENERIC PDIAGRAM AND SCORE FUNCTION --------------------
# ----------------------------------------------------------------------------- 
def ring_score(arg,
            argtype = None,
            flavour = 'geometric',
            base = None,
            nb_pers = np.inf,
            verbose = False,
            **kwargs):
    """Calculate pdiagram from given data structures.

    Generic function to calculate ring score by calling another appropriate
    ring score function:
        - If ``arg`` is a networkx graph, it will call ``ring_score_from_network``.
        - If ``arg`` is a ringity persistence diagram, it will call 
        ``ring_score_from_persistence_diagram``.
        - If ``arg`` is a numpy array or sparse matrix the paramter ``argtype`` 
        specifies how to interpret the data and which ring-score function to call.
        
    Note: All functions eventually call ``ring_score_from_sequence``.
    
    Parameters
    ----------
    arg : numpy array, sparse matrix, networkx graph or ringity persistence
    diagram.

    Returns
    -------
    score : float, ring-score of given object.
    """
    pdgm = pdiagram(arg,
            argtype = argtype,
            verbose = verbose,
            **kwargs)

    score = ring_score_from_pdiagram(
                                dgm = pdgm,
                                flavour = flavour,
                                base = base,
                                nb_pers = nb_pers)
    return score


def pdiagram(arg,
            argtype = None,
            verbose = False,
            **kwargs):
    """Calculate ring score from given data structures.

    Generic function to calculate persistence diagram by calling another appropriate
    pdiagram function:
        - If ``arg`` is a networkx graph, it will call ``pdiagram_from_network``.
        - If ``arg`` is a ringity persistence diagram, it will call 
        ``pdiagram_from_persistence_diagram``.
        - If ``arg`` is a numpy array or sparse matrix the paramter ``argtype`` 
        specifies how to interpret the data and which ring-score function to call. 
        ``None`` will try to guess weather the ``arg`` looks like a ``point cloud`` or a
        ``distance matrix``. 
        
    Note: All functions eventually call ``pdiagram_from_sequence``.
    
    Parameters
    ----------
    arg : numpy array, sparse matrix, networkx graph or ringity persistence
    diagram.

    Returns
    -------
    pdgm : PDiagram, persistence diagram of given object.
    """

    if isinstance(arg, nx.Graph):
        pdgm = pdiagram_from_network(arg)
    elif isinstance(arg, PDiagram):
        pdgm = arg
    elif isinstance(arg, np.ndarray) or isinstance(arg, scipy.sparse.spmatrix):
        if argtype is None:
            if is_valid_dm(arg):
                argtype = 'distance matrix'
            else:
                argtype = 'point cloud'
            if verbose:
                print(f"`argtype` was set to {argtype}")
        if argtype == 'point cloud':
            pdgm = pdiagram_from_point_cloud(arg, **kwargs)
        elif argtype == 'distance matrix':
            pdgm  = pdiagram_from_distance_matrix(arg, **kwargs)
        else:
            raise Exception(f"Argtype `{argtype} unknown")
    else:
        raise Exception

    return pdgm
                                                        
# -----------------------------------------------------------------------------    
# --------------------- DATA SPECIFIC RING SCORE FUNCTIONS --------------------
# -----------------------------------------------------------------------------     
def ring_score_from_point_cloud(X,
                            flavour = 'geometric',
                            base = None,
                            nb_pers = np.inf,
                            persistence = 'VietorisRipsPersistence',
                            metric = 'euclidean',
                            metric_params = {},
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
    pdgm = pdiagram_from_point_cloud(X,
                                persistence = persistence,
                                metric = metric,
                                metric_params = metric_params,
                                homology_dim = homology_dim,
                                **kwargs)
    return ring_score_from_pdiagram(pdgm,
                                flavour = flavour,
                                base = base,
                                nb_pers = nb_pers)
    
                                               
def ring_score_from_network(G,
                        flavour = 'geometric',
                        base = None,
                        nb_pers = np.inf,
                        persistence = 'VietorisRipsPersistence',
                        metric = 'net_flow',
                        metric_params = {},
                        homology_dim = 1,
                        **kwargs):
    """Calculate ring score from network.

    Parameters
    ----------
    G : networkx graph

    Returns
    -------
    score : float, ring-score of given object.
    """
    pdgm = pdiagram_from_network(G,
                            persistence = persistence,
                            metric = metric,
                            metric_params = metric_params,
                            homology_dim = homology_dim,
                            **kwargs)
    return ring_score_from_pdiagram(pdgm,
                                flavour = flavour,
                                base = base,
                                nb_pers = nb_pers)
                                    
                                    
def ring_score_from_distance_matrix(D,
                                persistence = 'VietorisRipsPersistence',
                                homology_dim = 1,
                                nb_pers = np.inf,
                                base = None,
                                flavour = 'geometric',
                                **kwargs):
    """Calculates ring-score from a PDiagram object.

    Parameters
    ----------
    D : _type_
        _description_
    persistence : str, optional
        _description_, by default 'VietorisRipsPersistence'
    homology_dim : int, optional
        _description_, by default 1
    nb_pers : _type_, optional
        _description_, by default np.inf
    base : _type_, optional
        _description_, by default None
    flavour : str, optional
        _description_, by default 'geometric'

    Returns
    -------
    _type_
        _description_
    """                                
    
    dgm = pdiagram_from_distance_matrix(D,
                                    persistence = persistence,
                                    homology_dim = homology_dim,
                                    **kwargs)
    return ring_score_from_pdiagram(dgm,
                                    flavour = flavour,
                                    base = base,
                                    nb_pers = nb_pers)
                            
                                    
def ring_score_from_pdiagram(dgm,
                             flavour = 'geometric',
                             nb_pers = np.inf,
                             base = None):
    """Calculates ring-score from a PDiagram object."""
    return ring_score_from_sequence(dgm.sequence,
                                    flavour = flavour,
                                    nb_pers = nb_pers,
                                    base = None)
                                               
# -----------------------------------------------------------------------------    
# ----------------------- PERSISTENCE DIAGRAM FUNCTIONS -----------------------
# ----------------------------------------------------------------------------- 
def pdiagram_from_point_cloud(X,
                              persistence = 'VietorisRipsPersistence',
                              metric = 'euclidean',
                              metric_params = {},
                              homology_dim = 1,
                              **kwargs):
    """Constructs a PDiagram object from a point cloud.
    
    This function wraps persistent homolgy calculation from giotto-tda."""
    
    
    if persistence == 'VietorisRipsPersistence':
        VR = VietorisRipsPersistence(metric = metric,
                                     metric_params = metric_params,
                                     homology_dimensions = tuple(range(homology_dim+1)),
                                     **kwargs)
        dgm = VR.fit_transform([X])[0]
    return PDiagram.from_gtda(dgm, homology_dim = homology_dim)   
    
    
def pdiagram_from_network(G, 
                        metric = 'net_flow', 
                        use_weights = None,
                        store_weights = None,
                        new_weight_name = None,
                        overwrite_weights = False,
                        verbose = False,
                        **kwargs):
    """Constructs a PDiagram object from a networkx graph.

    This function is not available yet. NEEDS TESTING!!!! 

    Parameters
    ----------
    G : _type_
        _description_
    metric : str, optional
        _description_, by default 'net_flow'
    use_weights : bool, optional
        _description_, by default None
    store_weights : bool, optional
        Weights will only be stored if a centrality measure is 
        used for distance calculations! By default None.
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

    D = pwdistance_from_network(G, metric = metric, verbose = verbose)
    return pdiagram_from_distance_matrix(D)
    
def pdiagram_from_distance_matrix(D, 
                                persistence = 'VietorisRipsPersistence',
                                homology_dim = 1,
                                **kwargs):
    """Constructs a PDiagram object from a distance matrix.
    """
    if persistence == 'VietorisRipsPersistence':
        VR = VietorisRipsPersistence(metric = "precomputed",
                                     homology_dimensions = tuple(range(homology_dim+1)),
                                     **kwargs)
        dgm = VR.fit_transform([D])[0]
    return PDiagram.from_gtda(dgm, homology_dim = homology_dim)