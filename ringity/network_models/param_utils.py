import numpy as np

from warnings import warn
from ringity.network_models.defaults import DEFAULT_RESPONSE_PARAMETER

def get_response_parameter(a, alpha, r):
    if a is None and alpha is None and r is None:
        return DEFAULT_RESPONSE_PARAMETER
    elif a is not None:
        warn("Using the parameter `a` for the response length is depricated. "
             "Please use the paramter `r` instead!",
             DeprecationWarning,
             stacklevel=2)
        return a
    elif alpha is not None:
        warn("Using the parameter `alpha` for the response length is depricated. "
             "Please use the paramter `r` instead!",
             DeprecationWarning,
             stacklevel=2)
        return alpha
    elif r is not None:
        return r
    else:
        return ValueError("Unknown error. Please contact the developers "
                          "if you encounter this in the wild.")

def get_rate_parameter(rate, beta):
    if beta is None and rate is None:
        raise ValueError(f"Please provide a distribution parameter: "
                         f"beta = {beta}, rate = {rate}")
    elif beta is not None and rate is not None:
        raise ValueError(f"Conflicting distribution parameters given: "
                         f"beta = {beta}, rate = {rate}")
    elif rate is not None:
        return rate
    elif beta is not None:
        if np.isclose(beta,0):
            return np.inf
        return np.tan(np.pi * (1-beta) / 2)
    else:
        return ValueError("Unknown error. Please contact the developers "
                          "if you encounter this in the wild.")

def get_coupling_parameter(K, rho, c, rate, r):
    if rho is None and K is None and c is None:
        raise ValueError(f"Please provide a coupling parameter: "
                         f"rho = {rho}, K = {K}")
    elif rho is not None and K is not None:
        raise ValueError(f"Conflicting coupling parameters given: "
                         f"rho = {rho}, K = {K}")
    elif rho is not None:
        return density_to_coupling_strength(rho=rho, r=r, rate=rate)
    elif K is not None:
        warn("Using the parameter `K` for the coupling strength is depricated. "
             "Please use the paramter `c` instead!",
             DeprecationWarning,
             stacklevel=2)
        return K
    elif c is not None:
        return c
    else:
        return ValueError("Unknown error. Please contact the developers "
                          "if you encounter this in the wild.")
                          
def get_response_parameter_from_density(rho, rate, c):
    if np.isclose(rate, 0):
        return rho / c

    if rate > 200:
        return None

    def density_constraint(r):
        
        if r == 0:
            return -rho
        
        plamb = np.pi * rate

        # Alternatively:
        # numerator = 2*(np.sinh(plamb*(1-a)) * np.sinh(plamb*a))
        numerator = (np.cosh(plamb) - np.cosh(plamb*(1 - r*2)))
        denominator = (r*2*plamb * np.sinh(plamb))

        return c - rho - c * numerator / denominator
    
    return bisect(density_constraint, 0, 1)
    
def mean_similarity(rate, response):
    """Calculate expected mean similarity (equivalently maximal expected density).

    This function assumes a (wrapped) exponential function and a cosine similarity
    of box functions.
    """

    if np.isclose(rate, 0):
        return response
    
    # Python can't handle this case properly; 
    #maybe the analytical expression can be simplified...
    if rate > 200:
        return 1.

    plamb = np.pi * rate

    # Alternatively:
    # numerator = 2*(np.sinh(plamb*(1-a)) * np.sinh(plamb*a))
    numerator = (np.cosh(plamb) - np.cosh(plamb*(1 - response*2)))
    denominator = (response*2*plamb * np.sinh(plamb))

    return 1 - numerator / denominator


def density_to_coupling_strength(rho, a = None, alpha = None, rate = None, beta = None, r = None):

    r = get_response_parameter(a=a, alpha=alpha, r=r)
    rate = get_rate_parameter(rate = rate, beta = beta)

    mu_S = mean_similarity(rate = rate, response = r)
    if rho <= mu_S:
        return rho/mu_S
    else:
        raise ValueError("Please provide a lower density!")

def coupling_strength_to_density(K = None, a = None, c = None, r = None, rate = None, beta = None, alpha = None):
    r = get_response_parameter(a=a, alpha=alpha, r=r)
    rate = get_rate_parameter(rate = rate, beta = beta)
    c = get_coupling_parameter(K=K, rho=None, c=c, rate=rate, r=r)

    rho = mean_similarity(response=r, rate = rate) * c
    return rho