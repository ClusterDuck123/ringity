import scipy.stats as ss

from ringity.classes._distns import _get_frozen_random_variable
from ringity.classes.diagram import PDgmPt, PDgm

def random_pdiagram_point(distn_arg = 'uniform', **kwargs):
    frozen_rv = _get_frozen_random_variable(distn_arg, **kwargs)
    return PDgmPt(sorted(frozen_rv.rvs(2)))
    
def random_pdiagram(size, distn_arg = 'uniform', **kwargs):
    return PDgm([random_pdiagram_point(distn_arg, **kwargs) for _ in range(size)])