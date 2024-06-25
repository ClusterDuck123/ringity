import numpy as np
import scipy.stats as ss

from scipy.optimize import newton
from scipy.spatial.distance import pdist

""" Utility module with functions leading up to the interaction probability function.

The general philosophy of the network model is a series of transformations 
after a given set of distances defined via: distances -> similarities -> probabilities.
"""


# =============================================================================
#  ------------------ NETWORK PARAMETER TRANSFORMATIONS ---------------------
# =============================================================================
@np.vectorize
def _get_rate(beta, rate):
    if beta is not None:
        rate = beta_to_rate(beta)
    return rate


def rate_to_beta(rate):
    beta = 1 - 2 / np.pi * np.arctan(rate)
    return beta


def beta_to_rate(beta):
    if np.isclose(beta, 0):
        rate = np.inf
    else:
        rate = np.tan(np.pi * (1 - beta) / 2)
    return rate


#  ----------------------------- ACTIVITY FUNCTIONS ---------------------------


def local_avg_activity(theta, r, beta=None, rate=None):
    rate = _get_rate(beta=beta, rate=rate)
    theta = theta % (2 * np.pi)

    Z = -np.expm1(-2 * np.pi * rate)

    A = np.exp(-rate * (theta - 2 * np.pi * r))
    B = Z + np.exp(-rate * (theta + 2 * np.pi * (1 - r)))
    value = np.where(theta >= 2 * np.pi * r, A, B) - np.exp(-rate * theta)

    return value / Z


#  ----------------------------- DENSITY FUNCTIONS ---------------------------


def global_coupling(rho, r, beta=None, rate=None) -> float:
    rate = _get_rate(beta=beta, rate=rate)

    numerator = np.pi * r * rate * np.sinh(rate * np.pi)
    denominator = np.pi * r * rate * np.sinh(rate * np.pi) - np.sinh(
        rate * np.pi * r
    ) * np.sinh(rate * np.pi * (1 - r))
    return rho * numerator / denominator


def global_response(rho, c, beta=None, rate=None) -> float:
    rate = _get_rate(beta=beta, rate=rate)

    def f_response(r):
        numerator = np.sinh(rate * np.pi * r) * np.sinh(rate * np.pi * (1 - r))
        denominator = r
        constant = np.pi * rate * np.sinh(rate * np.pi) * (1 - rho / c)

        return numerator / denominator - constant

    def fprime_response(r):
        numerator1 = np.sinh(np.pi * (1 - 2 * r) * rate)
        numerator2 = -np.sinh(np.pi * (1 - r) * rate) * np.sinh(np.pi * r * rate)
        denominator = r**2
        return ((np.pi * r * rate) * numerator1 + numerator2) / denominator

    # Only adds computational speed if initial guess is way off
    def fprime2_response(r):
        numerator1 = -np.cosh(np.pi * rate * (1 - 2 * r))
        numerator2 = -np.sinh(np.pi * rate * (1 - 2 * r))
        numerator3 = np.sinh(np.pi * rate * (1 - r)) * np.sinh(np.pi * r * rate)
        denominator = r**3
        return (
            2
            * (
                (np.pi * r * rate) ** 2 * numerator1
                + (np.pi * rate * r) * numerator2
                + numerator3
            )
            / denominator
        )

    # Initial guess, based on approximation formula
    r0 = np.clip(1 - (c - rho) / (c * beta), 0.001, 0.999)

    return newton(
        func=f_response, x0=r0, fprime=fprime_response, fprime2=fprime2_response
    )


def global_density(r, c, beta=None, rate=None) -> float:

    rate = _get_rate(beta=beta, rate=rate)
    if np.isinf(rate):
        return c

    if np.isclose(rate, 0):
        return r * c

    numerator = np.cosh(rate * np.pi) - np.cosh(rate * np.pi * (1 - 2 * r))
    denominator = 2 * np.pi * r * rate * np.sinh(rate * np.pi)
    return c * (1 - numerator / denominator)


def local_density(theta, r, c, beta=None, rate=None) -> float:
    """
    Local density function"""

    rate = _get_rate(beta=beta, rate=rate)
    if np.isinf(rate):
        return c

    if np.isclose(rate, 0):
        return r * c

    l = 2 * np.pi * r  # Response length
    s_min = np.clip((2 * r - 1) / r, 0, 1)

    theta0 = np.pi - abs(theta - np.pi)
    eps = np.sign(theta - np.pi)

    m = l * (1 - s_min)
    I1 = np.cosh(rate * m)
    I2a = np.exp(eps * rate * np.pi) * np.cosh(rate * np.pi - rate * l)
    I2b = (
        (rate * (m - theta0) - eps)
        * np.exp(eps * rate * (np.pi - theta0))
        * np.sinh(rate * np.pi)
    )
    values = 1 - np.where(m <= theta0, I1, I2a + I2b)
    normalization = np.exp(eps * rate * (theta0 - np.pi)) / (
        rate * l * np.sinh(np.pi * rate)
    )

    density = s_min - normalization * values

    return c * density


def second_moment_density(r, c, beta=None, rate=None) -> float:
    """
    (Continuous part of the) probability density function of s_r ∘ d(X,theta),
    where X is a wrapped exponentially distributed random variable
    with delay parameter beta, d(-,-) denotes the circular distance and s_r is the
    (normalized) area of the overlap of two boxes of length 2*pi*r on the circle.
    Normalization is taken to be the area of one box, 2*pi*r. Hence, the
    support is on [s_min,1], where s_min=|1 - (1-r)/r|_+.
    """
    l = 2 * np.pi * r  # Response length
    s_min = np.clip(1 - (1 - r) / r, 0, 1)

    rate = _get_rate(beta=beta, rate=rate)
    m = l * (1 - s_min)

    T1 = (
        np.cosh(2 * np.pi * rate - 3 * m * rate)
        - 9 * np.cosh(2 * np.pi * rate - m * rate)
        + 8 * np.cosh(2 * np.pi * rate)
    )
    T2 = (1 + s_min) * (
        np.sinh(2 * np.pi * rate - m * rate) + np.sinh(m * rate)
    ) - 2 * np.sinh(2 * np.pi * rate)
    T3 = (1 + s_min) * (1 - s_min) * (np.cosh(2 * np.pi * rate) - 1)

    mu2_s = s_min**2 + (T1 / (6 * rate**2 * l**2) + T2 / (l * rate) + T3) / (
        2 * np.sinh(-np.pi * rate) ** 2
    )
    return c**2 * mu2_s


# =============================================================================
#  ---------------------- STRING-TO-TRAFO FUNCTIONS -------------------------
# =============================================================================


def string_to_distribution(distn_name, **kwargs):
    """Take a string and returns a scipy.stats frozen random variable.

    Parameters to the frozen random variable can be passed on via ``**kwargs``
    (e.g. loc = 5 or scale = 3).
    The string is trying to match a corresponding scipy.stats class even
    if there is no exact match. For example, the string 'exponential' will
    return the random variable scipy.stats.expon.
    See function ``get_canonical_distn_name`` for details on name conversion.
    """

    # Check for edge cases
    if ("scale", np.inf) in kwargs.items():
        return ss.uniform(scale=2 * np.pi)

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
    distn_name = "".join(filter(str.isalnum, distn_name.lower()))
    distn_name_converer = {
        "normal": "norm",
        "exponential": "expon",
        "wrappedexponential": "wrappedexpon",
    }
    return distn_name_converer.get(distn_name, distn_name)


def get_canonical_sim_name(sim_name):
    """Transform string to match name of a similarity function."""
    sim_name = sim_name.lower()
    if not sim_name.endswith("_similarity"):
        sim_name = sim_name + "_similarity"
    return sim_name


def get_canonical_prob_name(prob_name):
    """Transform string to match name of a probability function."""
    prob_name = prob_name.lower()
    if not prob_name.endswith("_probability"):
        prob_name = prob_name + "_probability"
    return prob_name


# =============================================================================
#  --------------------------- TRANSFORMATIONS ------------------------------
# =============================================================================


def pw_angular_distance(x, metric="euclidean"):
    abs_dists = pdist(x.reshape(-1, 1), metric=metric)
    circ_dists = np.pi - abs(abs_dists - np.pi)
    return circ_dists


def angular_distance(x, y):
    abs_dist = abs(x - y)
    circ_dist = np.pi - abs(abs_dist - np.pi)
    return circ_dist


def box_cosine_similarity(dist, box_length):
    """Calculate the normalized overlap of two boxes on the circle."""

    if box_length == 0:
        return np.where(dist == 0, 1, 0)

    x1 = np.where(dist > box_length, 0, box_length - dist)
    # For box lengths larger than half of the circle there is a second overlap
    x2 = np.where(dist < 2 * np.pi - box_length, 0, box_length - (2 * np.pi - dist))

    total_overlap = x1 + x2
    return total_overlap / box_length


def response(x, theta, r=0.5):
    window_size = 2 * np.pi * r
    y = np.where(abs((x - theta) % (2 * np.pi)) < window_size, 1, 0)
    return y


def linear_probability(sims, slope, intercept):
    """Calculate linear transformation, capped at 0 an 1."""

    if slope < 0:
        Warning(f"Slope parameter is negative: {slope}")
    if intercept < 0:
        Warning(f"Intercept parameter is negative {intercept}")

    return (slope * sims + intercept).clip(0, 1)


def interaction_probability(d, r, c):
    l = 2 * np.pi * r  # Response length
    s_min = 1 - np.clip((1 - r) / r, 0, 1)  # Minimal similarity
    s = np.clip(1 - s_min - d / l, 0, 1) + s_min  # Similarity
    return c * s


# =============================================================================
#  --------------------------------- 3H TRAFOS ------------------------------
# =============================================================================


def hysteresis(r, beta=None, rate=None) -> float:
    rate = _get_rate(beta=beta, rate=rate)
    numer = np.sinh(np.pi * rate * r)
    denom = np.exp(np.pi * rate * (1 - r)) * np.sinh(np.pi * rate)
    return numer / denom


def heterogeneity(r, beta=None, rate=None) -> float:
    rate = _get_rate(beta=beta, rate=rate)
    numer = np.sinh(np.pi * rate * r) * np.sinh(np.pi * rate * (1 - r))
    denom = np.pi * r * rate * np.sinh(rate * np.pi)
    return numer / denom


def homophily(r, c) -> float:
    return 2 * np.arctan(c / (2 * np.pi * r)) / np.pi
