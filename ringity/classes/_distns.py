import inspect
import numpy as np
import scipy.stats as ss
import scipy.special as sc

from scipy.stats._distn_infrastructure import rv_generic, rv_frozen
from ringity.classes.exceptions import DistributionParameterError

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
        
def _get_rv(distn, **kwargs):
    
    if   isinstance(distn, str):
        if   distn == 'wrappedexpon':
            possible_args = inspect.getfullargspec(wrappedexpon._parse_args)[0]
            given_args = {key:value for (key,value) in kwargs.items() if key in possible_args}
            return wrappedexpon, given_args
        else:
            assert False, f"Distribution {distn} not known."
    
    elif isinstance(distn, rv_frozen):
        return ss.rv_continuous(name = 'frozen'), None
    else:
        assert False, f"data type of distn recognized: ({type(distn)})"

def get_rate_parameter(parameter, parameter_type):
    if   parameter_type.lower() == 'rate':
        return parameter
    elif parameter_type.lower() == 'shape':
        return 1/parameter
    elif parameter_type.lower() == 'delay':
        return np.cos(np.pi*parameter/2) / np.sin(np.pi*parameter/2)
    else:
        assert False, f"Parameter type '{parameter_type}' not known! " \
                       "Please choose between 'rate', 'shape' and 'delay'."