from ripser import ripser
from ringity.classes import Dgm
from ringity.centralities import net_flow
from ringity.methods import _yes_or_no
from ringity.constants import _assertion_statement
from ringity.exceptions import UnknownGraphType

import matplotlib
import networkx as nx
import numpy as np
import subprocess
import time
import os


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


def diagram(graph      = None,
            verbose    = False,
            use_weight = 'net_flow',
            distance_matrix = False,
            p          = 1):
    """
    Return the p-persistence diagram of an index- or distance-matrix.
    enforce_weights
    """

    if distance_matrix:
        D = graph
    else:
        input_type = type(graph)
        if input_type == np.ndarray:
            G = nx.from_numpy_array(graph)
            if use_weight is True:
                use_weight = 'weight'
        elif input_type == nx.classes.graph.Graph:
            G = graph
        else:
            raise UnknownGraphType(f"Unknown graph type {input_type}!")

        if not nx.get_edge_attributes(G, use_weight):
            induce_weight(G, weight=use_weight, verbose=verbose)

        if nx.number_of_selfloops(G) > 0:
            if verbose:
                print('Self loops in graph detected. They will be removed!')
            G.remove_edges_from(nx.selfloop_edges(G))

        t1 = time.time()
        D  = nx.floyd_warshall_numpy(G, weight=use_weight).A
        t2 = time.time()

        if verbose:
            print(f'Time for SPL calculation: {t2-t1}sec')

    t1 = time.time()
    ripser_output = ripser(np.array(D), maxdim=p, distance_matrix=True)
    dgm = Dgm(ripser_output['dgms'][p])
    t2 = time.time()

    if verbose:
        print(f'Time for ripser calculation: {t2-t1}sec')

    return dgm


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
        if exit_status: return exit_status

    if weight == 'net_flow':
        bb = net_flow(G)
        nx.set_edge_attributes(G, values=bb, name=weight)
    else:
        raise Exception(f"Weight '{weight}' unknown!")



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
