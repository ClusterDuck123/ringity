import scipy.stats as ss

from ringity.classes._distns import _get_frozen_random_variable
from ringity.classes.diagram import PersistenceDiagramPoint, PersistenceDiagram

def random_pdgm_point(distn_arg = 'uniform', **kwargs):
    frozen_rv = _get_frozen_random_variable(distn_arg, **kwargs)
    return PersistenceDiagramPoint(sorted(frozen_rv.rvs(2)))
    
def random_pdgm(length, distn_arg = 'uniform', **kwargs):
    return PersistenceDiagram([random_pdgm_point(distn_arg, **kwargs) for _ in range(length)])