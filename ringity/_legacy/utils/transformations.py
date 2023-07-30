import numpy as np
import scipy.stats as ss

from ringity.generators.utils.defaults import DEFAULT_RESPONSE_PARAMETER

""" Utility module with functions leading up to the interaction probability function.

The general philosophy of the network model is a series of transformations 
after a given set of distances defined via: distances -> similarities -> probabilities.
"""

# =============================================================================
#  ---------------------- STRING-TO-TRAFO FUNCTIONS -------------------------
# =============================================================================

def string_to_distribution(distn_name,
                           **kwargs):
    """Take a string and returns a scipy.stats frozen random variable.

    Parameters to the frozen random variable can be passed on via ``**kwargs``
    (e.g. loc = 5 or scale = 3).
    The string is trying to match a corresponding scipy.stats class even
    if there is no exact match. For example, the string 'exponential' will
    return the random variable scipy.stats.expon.
    See function ``get_canonical_distn_name`` for details on name conversion.
    """

    # Check for edge cases
    if ('scale', np.inf) in kwargs.items():
        return ss.uniform(scale = 2*np.pi)

    distn_name = get_canonical_distn_name(distn_name)
    try:
        rv_gen = getattr(ss, distn_name)
    except AttributeError:
        raise AttributeError(f"distribution {distn_name} unknown.")

    return rv_gen(**kwargs)


def string_to_similarity_function(sim_name):
    """Take a string and returns a similarity function."""
    sim_name = get_canonical_sim_name(sim_name)
    return eval(sim_name)


def string_to_probability_function(prob_name):
    """Take a string and returns a similarity function."""
    prob_name = get_canonical_prob_name(prob_name)
    return eval(prob_name)

#  ---------------- GET-CANONICAL-NAMES FUNCTIONS -------------------    

def get_canonical_distn_name(distn_name):
    """Transform string to match a corresponding scipy.stats class."""
    distn_name = ''.join(filter(str.isalnum, distn_name.lower()))
    distn_name_converer = {
        'normal' : 'norm',
        'exponential' : 'expon',
        'wrappedexponential' : 'wrappedexpon'
    }
    return distn_name_converer.get(distn_name, distn_name)


def get_canonical_sim_name(sim_name):
    """Transform string to match name of a similarity function."""
    sim_name = sim_name.lower()
    if not sim_name.endswith('_similarity'):
        sim_name = sim_name + '_similarity'
    return sim_name


def get_canonical_prob_name(prob_name):
    """Transform string to match name of a probability function."""
    prob_name = prob_name.lower()
    if not prob_name.endswith('_probability'):
        prob_name = prob_name + '_probability'
    return prob_name


# =============================================================================
#  --------------------------- TRANSFORMATIONS ------------------------------
# =============================================================================

def box_cosine_similarity(dists, box_length):
    """Calculate the normalized overlap of two boxes on the circle."""
    
    if box_length == 0:
        return np.where(dists == 0, 1, 0)
        
    x1 = (box_length - dists).clip(0)
    # For box lengths larger than half of the circle there is a second overlap
    if box_length <= np.pi:
        x2 = 0
    else:
        x2 = (dists - (2*np.pi - box_length)).clip(0)

    total_overlap = x1 + x2
    return total_overlap / box_length


def linear_probability(sims, slope, intercept):
    """Calculate linear transformation, capped at 0 an 1."""

    if slope < 0:
        Warning(f"Slope parameter is negative: {slope}")
    if intercept < 0:
        Warning(f"Intercept parameter is negative {intercept}")

    return (slope*sims + intercept).clip(0, 1)