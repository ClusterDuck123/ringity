from ripser import ripser
from ringity.classes import Dgm, DgmPt
from ringity.centralities import net_flow
from ringity.methods import dict2numpy, _yes_or_no
from ringity.constants import _assertion_statement
from ringity.exceptions import DigraphError, UnknownGraphType

import matplotlib
import networkx as nx
import numpy as np
import subprocess
import time
import os


def get_distance_matrix(G, distance, verbose=False, spl_method='floyd_warshall'):
    if distance == 'induce':
        if verbose:
            print('net-flow distance will be induced.')
        distance = 'net-flow'
        induce_distance(G, name=distance, verbose=verbose)

    else:
        v,w = next(iter(G.edges))
        attributes = set(G[v][w])

        if not (distance or attributes):
            if verbose:
                print('No weights detected, net-flow distance will be induced.')
            distance = 'net-flow'
            induce_distance(G, name=distance, verbose=verbose)

        elif distance is None:
            distance = next(iter(attributes))
        elif distance in attributes:
            pass
        else:
            raise KeyError(f"No edge attribute called '{distance}' found!")

    if verbose:
            print(f"Using edge attribute '{distance}' as distance.")

    if spl_method == 'floyd_warshall':
        t1 = time.time()
        D  = nx.floyd_warshall_numpy(G, weight=distance)
        t2 = time.time()

        if verbose:
            print(f'Time for Floyd-Warshall calculation: {t2-t1}sec')
    else:
        # UNFINISHED !!!!
        raise Exception # GENERIC EXCEPTION RAISED HERE !!!!

    return D


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


def diagram(graph      = None ,
            distance   = None ,
            verbose    = False,
            induce     = False,
            p          = 1    ,
            spl_method = 'floyd_warshall' ,
            inplace    = True ):
    """Return the p-persistence diagram of an index- or distance-matrix."""

    if induce:
        distance = 'induce'

    input_type = type(graph)

    if input_type == np.ndarray:
        D = graph

    elif input_type == nx.classes.graph.Graph:
        G = graph
        pathology, dgm  = _pathological_cases(G=G, distance=distance, verbose=verbose)
        if pathology:
            return dgm

        D = get_distance_matrix(G          = G,
                                distance   = distance,
                                verbose    = verbose,
                                spl_method = spl_method)

    elif input_type == nx.classes.digraph.DiGraph:
        raise DigraphError('Graph is a digraph, please provide an undirected '
                           'graph!')
    else:
        raise UnknownGraphType(f'Unknown graph type {input_type}!')

    assert D is not None, _assertion_statement

    t1 = time.time()
    ripser_output = ripser(np.array(D), maxdim=p, distance_matrix=True)
    dgm = Dgm(ripser_output['dgms'][p])
    t2 = time.time()

    if verbose:
        print(f'Time for ripser calculation: {t2-t1}sec')
    return dgm



def induce_distance(G, name = 'distance', verbose=False):
    if verbose:
        v,w = next(iter(G.edges))
        attributes = G[v][w]

        if name in attributes:
            answer = input(
                    f"Graph has already an edge attribute called '{name}'! "
                    f"Do you want to overwrite this edge attribute? ")
            proceed = _yes_or_no(answer)
            if not proceed:
                print('Process terminated!')
                return

    if nx.number_of_selfloops(G) > 0:
            if verbose:
                print('Self loops in graph detected. They will be removed!')
            G.remove_edges_from(nx.selfloop_edges(G))

    bb = net_flow(G, verbose=verbose)
    nx.set_edge_attributes(G, values=bb, name=name)



def get_coordinates(dgm,
                    infinity = True):

    z = [(k.birth, k.death) for k in dgm if k.death != np.inf]
    x_max = max(k.birth for k in dgm)
    if z:
        x, y = map(list,zip(*z))
        y_max = max(max(y), x_max)
    else:
        x, y = [], []
        y_max = x_max


    z2 = [(k.birth, y_max+1) for k in dgm if k.death == np.inf]

    if z2 and infinity:
        x2, y2 = map(list,zip(*z2))
    else:
        x2, y2 = [], []
        for x_tmp, y_tmp in z2:
            x.append(x_tmp)
            y.append(y_tmp)

    return x, y , x2, y2



def draw_diagram(dgm,
                 line=None,
                 title=True,
                 infinity=True,
                 ax = None,
                 verbose = None):
    """Plot persistence diagram."""

    x,y,x2,y2 = get_coordinates(dgm, infinity=infinity)

    if not y:
        y_max = max(x + x2)
    else:
        y_max = max(max(y), max(x + x2))

    if x2 and infinity:
        fig, (ax2, ax) = plt.subplots(2, 1, sharex=True, gridspec_kw = {'height_ratios':[1, 10]})
    else:
        if ax is None:
            fig, ax = plt.subplots()

    if line is None:
        line = [0, y_max]


    ax.plot(x,y, '*', line, line)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.tick_bottom()
    ax.set_xlabel('Time of Birth')
    ax.set_ylabel('Time of Death')

    if x2 and infinity:
        ax.set_ylim(0, y_max)
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-0.015, 0.015), (1, 1), **kwargs)  # top of y-axis

        ax2.plot(x2,y2, '*');
        ax2.set_ylim(y_max+0.5, y_max+1.5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_yticks([])
        ax2.set_ylabel('inf', rotation=0, position=(0,0.25))

    if type(title) == str:
        pass
        #ax2.title(title)
    else:
        pass
        #ax2.title('max. persistence = %.3f'%p_max)

    if verbose:
        return fig
