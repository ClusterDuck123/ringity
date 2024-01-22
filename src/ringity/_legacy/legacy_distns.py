import scipy.stats as ss

from scipy.stats._distn_infrastructure import rv_generic, rv_frozen

"""This module is deprecated and should be safely removed in the future."""


# =============================================================================
#  -------------------------- GLOBAL CONSTANTS ------------------------------
# =============================================================================

DISTN_NAME2LEMMA = {
    'normal' : 'norm',
    'exponential' : 'expon',
    'wrappedexponential' : 'wrappedexpon'
}

RINGITY_DISTN_NAMES_LIST = ['wrappedexpon']
SCIPY_DISTN_NAMES_LIST = list(filter
                (
                lambda d: isinstance(getattr(ss, d), rv_generic),
                dir(ss))
                )

# =============================================================================
#  ------------------------------ FUNCTIONS ----------------------------------
# =============================================================================

def _get_frozen_random_variable(distn_arg, **kwargs):
    if isinstance(distn_arg, rv_frozen):
        frozen_rv = distn_arg
    else:
        if isinstance(distn_arg, str):
            distn_name = DISTN_NAME2LEMMA.get(distn_arg, distn_arg)
            if distn_name in SCIPY_DISTN_NAMES_LIST:
                rv_gen = getattr(ss, distn_name)
            elif distn_name in RINGITY_DISTN_NAMES_LIST:
                rv_gen = eval(distn_name)
        elif isinstance(distn_arg, rv_generic):
            rv_gen = distn_arg
        else:
            raise ValueError(f'Type {type(distn_arg)} unknown.')
        frozen_rv = rv_gen(**kwargs)
        
    assert isinstance(frozen_rv, rv_frozen)
    return frozen_rv