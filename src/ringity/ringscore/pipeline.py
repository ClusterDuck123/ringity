from ringity.ringscore.metric2ringscore import ring_score_from_sequence
from ringity.ringscore.data2metric import (
    pwdistance_from_network,
    pwdistance_from_point_cloud,
)
from ringity.tda.pdiagram.pdiagram import PDiagram
from ringity.tda import phomology

from scipy.spatial.distance import is_valid_dm

import numpy as np
import networkx as nx
import scipy.sparse


# -----------------------------------------------------------------------------
# -------------------- GENERIC PDIAGRAM AND SCORE FUNCTION --------------------
# -----------------------------------------------------------------------------
def ring_score(
    arg,
    argtype=None,
    flavour="geometric",
    exponent=2,
    nb_pers=np.inf,
    verbose=False,
    **kwargs,
):
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
    pdgm = pdiagram(arg, argtype=argtype, verbose=verbose, **kwargs)

    score = ring_score_from_pdiagram(
        pdgm=pdgm, flavour=flavour, exponent=exponent, nb_pers=nb_pers
    )
    return score


def pdiagram(arg, argtype=None, verbose=False, **kwargs):
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
                argtype = "distance matrix"
            else:
                argtype = "point cloud"
            if verbose:
                print(f"`argtype` was set to {argtype}")
        if argtype == "point cloud":
            pdgm = pdiagram_from_point_cloud(arg, **kwargs)
        elif argtype == "distance matrix":
            pdgm = pdiagram_from_distance_matrix(arg, **kwargs)
        else:
            raise Exception(f"Argtype `{argtype} unknown.")
    else:
        raise Exception(f"Data structure `{type(arg)} unknown.")

    return pdgm


# -----------------------------------------------------------------------------
# --------------------- DATA SPECIFIC RING SCORE FUNCTIONS --------------------
# -----------------------------------------------------------------------------
def ring_score_from_point_cloud(
    X,
    flavour="geometric",
    exponent=2,
    nb_pers=np.inf,
    persistence="VietorisRipsPersistence",
    metric="euclidean",
    metric_params={},
    dim=1,
    **kwargs,
):
    """Calculate ring score from point cloud.

    Parameters
    ----------
    X : numpy array, sparse matrix

    Returns
    -------
    score : float, ring-score of given object.
    """
    pdgm = pdiagram_from_point_cloud(
        X,
        persistence=persistence,
        metric=metric,
        metric_params=metric_params,
        dim=dim,
        **kwargs,
    )
    return ring_score_from_pdiagram(
        pdgm, flavour=flavour, exponent=exponent, nb_pers=nb_pers
    )


def ring_score_from_network(
    G,
    flavour="geometric",
    exponent=2,
    nb_pers=np.inf,
    persistence="VietorisRipsPersistence",
    metric="net_flow",
    metric_params={},
    dim=1,
    use_weights=None,
    **kwargs,
):
    """Calculate ring score from network.

    Parameters
    ----------
    G : networkx graph

    Returns
    -------
    score : float, ring-score of given object.
    """
    pdgm = pdiagram_from_network(
        G,
        persistence=persistence,
        metric=metric,
        metric_params=metric_params,
        dim=dim,
        use_weights=None,
        **kwargs,
    )
    return ring_score_from_pdiagram(
        pdgm, flavour=flavour, exponent=exponent, nb_pers=nb_pers
    )


def ring_score_from_distance_matrix(
    D,
    persistence="VietorisRipsPersistence",
    dim=1,
    nb_pers=np.inf,
    exponent=2,
    flavour="geometric",
    **kwargs,
):
    """Calculates ring-score from a PDiagram object.

    Parameters
    ----------
    D : _type_
        _description_
    persistence : str, optional
        _description_, by default 'VietorisRipsPersistence'
    dim : int, optional
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

    pdgm = pdiagram_from_distance_matrix(D, persistence=persistence, dim=dim, **kwargs)
    return ring_score_from_pdiagram(
        pdgm, flavour=flavour, exponent=exponent, nb_pers=nb_pers
    )


def ring_score_from_pdiagram(
    pdgm, score_type="length", flavour="geometric", nb_pers=np.inf, exponent=2
):
    """Calculates ring-score from a PDiagram object."""
    trimmed_pdgm = pdgm.trimmed(nb_pers)

    if score_type == "length":
        score = ring_score_from_sequence(
            trimmed_pdgm.plengths, flavour=flavour, nb_pers=nb_pers, exponent=exponent
        )
    elif score_type == "ratio":
        score = ring_score_from_sequence(
            trimmed_pdgm.pratios, flavour=flavour, nb_pers=nb_pers, exponent=exponent
        )
    elif score_type == "diameter":
        score = ring_score_from_sequence(
            trimmed_pdgm.plengths, flavour=flavour, nb_pers=nb_pers, exponent=exponent
        )
        score *= trimmed_pdgm.signal / trimmed_pdgm.diameter * (2 / np.sqrt(3))

    else:
        raise Exception(f"Score type `{score_type}` unknown.")

    return score


# -----------------------------------------------------------------------------
# ----------------------- PERSISTENCE DIAGRAM FUNCTIONS -----------------------
# -----------------------------------------------------------------------------
def pdiagram_from_point_cloud(
    X, metric="euclidean", dim=1, persistence="VietorisRipsPersistence", **kwargs
):
    """Constructs a PDiagram object from a point cloud.

    This function wraps persistent homolgy calculation from giotto-tda."""

    if persistence == "VietorisRipsPersistence":
        D = pwdistance_from_point_cloud(X, metric=metric)
        pdgm = pdiagram_from_distance_matrix(
            D, dim=dim, persistence=persistence, kwargs=kwargs
        )
    return pdgm


def pdiagram_from_network(
    G,
    metric="net_flow",
    use_weights=None,
    store_weights = True,
    remove_self_loops = True,
    verbose=False,
    **kwargs,
):
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
    verbose : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """

    D = pwdistance_from_network(
        G,
        metric=metric,
        verbose=verbose,
        use_weights=use_weights,
        store_weights=store_weights,
        remove_self_loops=remove_self_loops,
    )
    return pdiagram_from_distance_matrix(D)


def pdiagram_from_distance_matrix(
    D, persistence="VietorisRipsPersistence", dim=1, **kwargs
):
    pdgm = phomology.pdiagram_from_distance_matrix(
        D, persistence=persistence, dim=dim, **kwargs
    )
    return pdgm
