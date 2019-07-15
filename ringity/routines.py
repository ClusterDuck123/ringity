from .constants import _ringity_parameters, _assertion_statement

import numpy as np


# --------------------------------------------------------------------------
#
# THIS MODULE IS DEPRICATED AND SHOUD NOT BE PART OF THE PACKAGE ANYMORE !!!
#
# --------------------------------------------------------------------------



def _yes_or_no(answer):
    answer = answer.lower()
    while answer not in {*_ringity_parameters['yes'], *_ringity_parameters['no']}:
        answer = input("Please provide a readable input, e.g. 'y' or 'n'! ")
        
    if answer in _ringity_parameters['yes']:
        return True
    elif answer in _ringity_parameters['no']:
        return False
    else: 
        assert False, _assertion_statement



def dict2numpy(bb, fill=np.inf):
    """
    Returns a numpy array corresponding to the given dictionary and fills the 
    vacancies with fill.
    """
    
    nodes = bb.keys()
    N = len(nodes)
    D = np.full((N, N), np.inf)
    
    for v in nodes:
        for w in bb[v]:
            D[v,w] = bb[v][w]
    return D


def dict2ddict(bb):
    sources, targets = zip(*bb.keys())
    nodes = {*set(sources), *set(targets)}
    new_bb = {v:{
                **{s:bb[(s,v)] for s in sources if (s,v) in bb},
                **{t:bb[(v,t)] for t in targets if (v,t) in bb},
                } for v in nodes}

    return new_bb


def ddict2dict(bb):   
    edges = set.union(
            *( {frozenset({s,t}) for s in bb[t].keys() if s!=t}
                                            for t in bb.keys() )
                      )
    new_bb = {(s,t):bb[s][t] for (s,t) in edges}
    return new_bb