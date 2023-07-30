from ringity._legacy.legacy_distns import _get_frozen_random_variable
from ringity.tda.pdiagram.pdiagram import PDiagramPoint, PDiagram

def random_pdgm_point(distn_arg = 'uniform', **kwargs):
    frozen_rv = _get_frozen_random_variable(distn_arg, **kwargs)
    return PDiagramPoint(sorted(frozen_rv.rvs(2)))
    
def random_pdgm(length, distn_arg = 'uniform', **kwargs):
    return PDiagram([random_pdgm_point(distn_arg, **kwargs) for _ in range(length)])