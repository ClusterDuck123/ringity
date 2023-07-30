import numpy as np

from warnings import warn
from scipy.optimize import bisect
from ringity.generators.utils.defaults import DEFAULT_RESPONSE_PARAMETER

# =============================================================================
#  -------------------------- PARAMETER INFERENCE ----------------------------
# =============================================================================

def infer_response_parameter(density, coupling, rate):
    if density  > coupling:
        raise AttributeError(f"Density can't be higher than coupling. "
                             f"Please decrease density ({density}) or "
                             f"increase coupling ({coupling}).")
    if np.isclose(rate, 0):
        response = density / coupling
        
    elif rate > 200:
        response = np.nan
        
    else:
        def response_equation(r):
            return governing_equation(
                            response = r,
                            density = density,
                            coupling = coupling,
                            rate = rate
                            )
        response = bisect(response_equation, 0.0001, 1)

    return response
    
    
def infer_coupling_parameter(density, response, rate):
    mu_S = max_density(rate = rate, response = response)
    coupling = density/mu_S
    if coupling <= 1:
        return coupling
    else:
        raise ValueError("Please provide a lower density!")
            
def infer_density_parameter(response, rate, coupling):
    density = coupling * max_density(response = response, rate = rate)
    return density
        
def infer_rate_parameter(response, density, coupling):
    def rate_equation(lamb):
        return governing_equation(
                        response = response,
                        density = density,
                        coupling = coupling,
                        rate = lamb
                        )
    rate = bisect(rate_equation, 0, 200)
    return rate
        
def governing_equation(response, density, coupling, rate):
    return coupling * max_density(response = response, rate = rate) - density
        
def max_density(rate, response):
    """Calculate maximal allowed density (equivalently expected mean similarity).

    This function assumes a (wrapped) exponential function and a cosine similarity
    of box functions.
    """

    if np.isclose(rate, 0):
        return response
        
    if np.isclose(response, 0):
        return 0

    # Python can't handle this case properly;
    # maybe the analytical expression can be simplified...
    if rate > 200:
        return 1.

    plamb = np.pi * rate

    # Alternatively:
    # numerator = 2*(np.sinh(plamb*(1-a)) * np.sinh(plamb*a))
    numerator = (np.cosh(plamb) - np.cosh(plamb*(1 - response*2)))
    denominator = (response*2*plamb * np.sinh(plamb))

    return 1 - numerator / denominator
            
#  -------------------------- OLD INFERENCE FUNCTIONS ----------------------------

def denisty_to_response(density, coupling, response, rate):
    if np.isclose(rate, 0):
        return density / coupling

    if rate > 200:
        return None

    def density_constraint(r):
        if r == 0:
            return -density

        plamb = np.pi * rate

        # Alternatively:
        # numerator = 2*(np.sinh(plamb*(1-a)) * np.sinh(plamb*a))
        numerator = (np.cosh(plamb) - np.cosh(plamb*(1 - r*2)))
        denominator = (r*2*plamb * np.sinh(plamb))

        return coupling - density - coupling * numerator / denominator

    return bisect(density_constraint, 0, 1)


def density_to_coupling(density, response, rate):
    mu_S = mean_similarity(rate = rate, response = response)
    if rho <= mu_S:
        return density/mu_S
    else:
        raise ValueError("Please provide a lower density!")


def coupling_to_density(coupling, response, rate):
    density = coupling * mean_similarity(response = response, rate = rate)
    return density


def mean_similarity(rate, response):
    """Calculate expected mean similarity (equivalently maximal expected density).

    This function assumes a (wrapped) exponential function and a cosine similarity
    of box functions.
    """

    if np.isclose(rate, 0):
        return response

    # Python can't handle this case properly;
    # maybe the analytical expression can be simplified...
    if rate > 200:
        return 1.

    plamb = np.pi * rate

    # Alternatively:
    # numerator = 2*(np.sinh(plamb*(1-a)) * np.sinh(plamb*a))
    numerator = (np.cosh(plamb) - np.cosh(plamb*(1 - response*2)))
    denominator = (response*2*plamb * np.sinh(plamb))

    return 1 - numerator / denominator

# =============================================================================
#  --------------------------- PARAMETER PARSERS -----------------------------
# =============================================================================

def parse_response_parameter(r = None,
                             a = None,
                             alpha = None,
                             response = None):

    nb_parms = sum(1 for parm in (r, a, alpha, response) if parm is not None)

    if nb_parms == 0:
        response = None

    elif nb_parms > 1:
        raise ValueError(f"Please provide only one response parameter! " \
                         f"r = {r}, a = {a}, alpha = {alpha}, " \
                         f"response = {response}")

    elif a is not None:
        warning_message = "Using the parameter `a` for the response length " \
                          "is depricated. Please use the paramter `r` instead!"
        warn(warning_message, DeprecationWarning, stacklevel = 2)
        response = a

    elif alpha is not None:
        warning_message = "Using the parameter `alpha` for the response length " \
                         "is depricated. Please use the paramter `r` instead!"
        warn(warning_message, DeprecationWarning, stacklevel = 2)
        response = alpha

    elif r is not None:
        response = r

    elif response is not None:
        response = response

    else:
        return ValueError("Unknown error. Please contact the developers "
                          "if you encounter this in the wild.")
    return response


def parse_rate_parameter(rate, beta):
    if beta is None and rate is None:
        raise ValueError(f"Please provide a distribution parameter: "
                         f"beta = {beta}, rate = {rate}")

    elif beta is not None and rate is not None:
        raise ValueError(f"Conflicting distribution parameters given: "
                         f"beta = {beta}, rate = {rate}")

    elif beta is not None:
        rate = beta_to_rate(beta)
        
    elif rate is not None:
        rate = rate

    else:
        return ValueError("Unknown error. Please contact the developers "
                          "if you encounter this in the wild.")
    return rate


def parse_coupling_parameter(c = None,
                             K = None,
                             coupling = None):

    nb_parms = sum(1 for parm in (c, K, coupling) if parm is not None)

    if nb_parms == 0:
        coupling = None

    elif nb_parms > 1:
        raise ValueError(f"Please provide only one coupling parameter! " \
                         f"r = {r}, a = {a}, alpha = {alpha}, " \
                         f"response = {response}")

    elif c is not None:
        coupling = c

    elif K is not None:
        warning_message = "Using the parameter `K` for the coupling strength " \
                          "is depricated. Please use the paramter `c` instead!"
        warn(warning_message, DeprecationWarning, stacklevel=2)
        coupling = K

    elif coupling is not None:
        coupling = coupling
    else:
        raise ValueError("Unknown error. Please contact the developers "
                          "if you encounter this in the wild.")
    return coupling
    
def parse_density_parameter(rho = None,
                            density = None):

    if rho is not None and density is not None:
        if rho != density:
            raise ValueError(f"Conflicting distribution parameters given: "
                             f"rho = {rho}, density = {density}")

    elif rho is not None:
        density = rho
    else:
        density = density

    return density
    
    
def parse_canonical_parameters(params):
    rate = parse_rate_parameter(
                            rate = params['rate'],
                            beta = params['beta'])
    response = parse_response_parameter(
                            r = params['r'],
                            a = params['a'],
                            alpha = params['alpha'],
                            response = params['response'])
    coupling = parse_coupling_parameter(
                            c = params['c'],
                            K = params['K'],
                            coupling = params['coupling'])
    density = parse_density_parameter(
                            rho = params['rho'],
                            density = params['density'])
    model_params = {
        'rate' : rate,
        'response' : response,
        'coupling' : coupling,
        'density': density}
    return model_params


def rate_to_beta(rate):
    beta = 1 - 2/np.pi * np.arctan(rate)
    return beta
    
def beta_to_rate(beta):
    if np.isclose(beta,0):
        rate = np.inf
    else:
        rate = np.tan(np.pi * (1-beta) / 2)
    return rate