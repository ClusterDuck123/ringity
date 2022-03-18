"""
Collection of functions for data structure conversions.
Some tests still rely on this module.
"""

def ddict2dict(bb):
    edges = set.union(
            *( {frozenset({s,t}) for s in bb[t].keys() if s!=t}
                                            for t in bb.keys() )
                      )
    new_bb = {(s,t):bb[s][t] for (s,t) in edges}
    return new_bb
    

def dict2ddict(bb):
    sources, targets = zip(*bb.keys())
    nodes = {*set(sources), *set(targets)}
    new_bb = {v:{
                **{s:bb[(s,v)] for s in sources if (s,v) in bb},
                **{t:bb[(v,t)] for t in targets if (v,t) in bb},
                } for v in nodes}

    return new_bb
