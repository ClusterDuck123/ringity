import numpy as np

from warnings import warn
from ringity.network_models.defaults import DEFAULT_RESPONSE_PARAMETER

# =============================================================================
#  -------------------------- PARAMETER INFERENCE ----------------------------
# =============================================================================

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
    #maybe the analytical expression can be simplified...
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
        if np.isclose(beta,0):
            rate = np.inf
        else:
            rate = np.tan(np.pi * (1-beta) / 2)

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
        return ValueError("Unknown error. Please contact the developers "
                          "if you encounter this in the wild.")
    return coupling