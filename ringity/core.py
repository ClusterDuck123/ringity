from ripser import ripser
from ringity.gtda import vietoris_rips_from_point_cloud
from ringity.centralities import net_flow, resistance
from ringity.classes.diagram import PersistenceDiagram
from ringity.readwrite.prompts import _yes_or_no, _assertion_statement
from ringity.classes.exceptions import UnknownGraphType

import os
import time
import subprocess
import scipy.sparse

import numpy as np
import networkx as nx


def ring_score_from_persistence_diagram(dgm, flavour = 'geometric'):
    if flavour == 'geometric':
        return dgm.ring_score
    elif flavour == 'amplitude':
        return dgm.sequence[0] / np.sum(dgm.sequence)
    elif flavour == 'entropy':
        # Needs to be checked / adapted
        if base is None:
            base = np.e

        # Normalize by uniform distribution on non-zero persistences.

        # dgm.trimmed() reduces the diagram to points with positive length.
        # This should usually be the case anyways but since this can have an effect on
        # entropy it's explicitly ensured.
        normalisation = np.log(len(dgm.trimmed()))
        return 1 - np.log(base)*ss.entropy(dgm.sequence, base = base) / normalisation
    else:
        raise Exception(f"Flavour {flavour} unknown.")

def ring_score(arg, argtype = 'pointcloud', flavour = 'geometric'):
    """Calculates ring score from various data structures and of different
    flavours.

    If X is a numpy array or sparse matrix the paramter ``argtype`` specifies
    how to interpret the data.

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
            dgm = vietoris_rips_from_point_cloud(arg)
        else:
            raise Error
    else:
        raise Error

    score = ring_score_from_persistence_diagram(dgm=dgm, flavour=flavour)
    return score


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
                                metric=metric,
                                verbose=verbose)

    t1 = time.time()
    ripser_output = ripser(np.array(D), maxdim=p, distance_matrix=True)
    dgm = PersistenceDiagram(ripser_output['dgms'][p])
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
