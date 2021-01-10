from ripser import ripser
from ringity.classes import Dgm
from ringity.centralities import net_flow, resistance
from ringity.methods import _yes_or_no
from ringity.constants import _assertion_statement
from ringity.exceptions import UnknownGraphType

import networkx as nx
import numpy as np
import subprocess
import time
import os

def diagram(graph = None,
            verbose = False,
            metric = 'net_flow',
            distance_matrix = False,
            efficiency='speed',
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
        D = graph
    else:
        input_type = type(graph)
        if input_type == np.ndarray:
            G = nx.from_numpy_array(graph)
        elif input_type == nx.classes.graph.Graph:
            G = graph
        else:
            raise UnknownGraphType(f"Unknown graph type {input_type}!")

        if nx.number_of_selfloops(G) > 0:
            if verbose:
                print('Self loops in graph detected. They will be removed!')
            G.remove_edges_from(nx.selfloop_edges(G))

        D = get_distance_matrix(G,
                                metric=metric,
                                efficiency=efficiency,
                                verbose=verbose)

    t1 = time.time()
    ripser_output = ripser(np.array(D), maxdim=p, distance_matrix=True)
    dgm = Dgm(ripser_output['dgms'][p])
    t2 = time.time()

    if verbose:
        print(f'Time for ripser calculation: {t2-t1:.6f} sec')

    return dgm


def get_distance_matrix(G, metric, efficiency='speed', verbose=False):
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
                          efficiency=efficiency,
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


def induce_weight(G, weight = 'net_flow', efficiency='speed', verbose=False):
    if verbose and nx.get_edge_attributes(G, weight):
        exit_status = existing_edge_attribute_warning(weight)
        if exit_status: return exit_status

    if   weight == 'net_flow':
        bb = net_flow(G, efficiency=efficiency)
    elif weight == 'betweenness':
        bb = nx.edge_betweenness_centrality(G)
    elif weight == 'current_flow':
        bb = nx.edge_current_flow_betweenness_centrality(G)
    else:
        raise Exception(f"Weight '{weight}' unknown!")

    nx.set_edge_attributes(G, values=bb, name=weight)


# ---------------------------- NOT IMPLEMENTED YET ----------------------------

def _pathological_cases(G, distance, verbose):
    E = G.number_of_edges()
    N = G.number_of_nodes()

    if N == 1:
        if verbose:
            print('Graph with only one node was given.')
        return True, Dgm()
    elif E == round(N*(N-1)/2) and not distance:
        if verbose:
            print('Complete graph with no edge attribute was given.')
        return True, Dgm()
    else:
        return False, None
