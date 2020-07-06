from ringity.distribution_functions import mean_similarity, cdf_similarity
from scipy.spatial.distance import pdist, squareform
from numpy import pi as PI

import scipy
import numpy as np
import networkx as nx

# =============================================================================
#  -------------------------------  PREPARATION -----------------------------
# =============================================================================
def geodesic_distances(thetas):
    abs_dists = pdist(thetas.reshape(-1,1))
    return np.where(abs_dists<PI, abs_dists, 2*PI-abs_dists)


def overlap(dist, a):
    """
    Calculates the overlap of two boxes of length 2*pi*a on the circle for a
    given distance dist (measured from center to center).
    """

    x1 = (2*PI*a-dist).clip(0)
    if a <= 0.5:
        return x1
    # for box sizesw with a>0 there is a second overlap
    else:
        x2 = (dist-2*PI*(1-a)).clip(0)
        return x1 + x2


def slope(rho, kappa, a):
    mu_S = mean_similarity(kappa,a)
    if rho <= mu_S:
        return rho/mu_S
    else:
        const = 1/np.sinh(PI*kappa)
        def integral(k):
            term1 = np.sinh((1 + 2*a*(1/k-1))*PI*kappa)
            term2 = (k*np.sinh((a*PI*kappa)/k)*np.sinh(((a+k-2*a*k)*PI*kappa)/k))/(a*PI*kappa)
            return term1-term2
        return scipy.optimize.newton(
            func = lambda k: const*integral(k) + (1-cdf_similarity(1/k, kappa, a)) - rho,
            x0 = rho/mu_S)

# =============================================================================
#  ------------------------------  NETWORK MODEL ----------------------------
# =============================================================================

def weighted_network_model(N, rho, a=0.5, kappa=0):
    """
    Returns samples of the Network model as described in [1]
    The outputs are samples of the
     - positions of the nodes placed on the circle according to a von
       Mises distribution,
     - their pairwise distances
     - their similarities, given an 'activity window' of size a
     - their connection probabilities, given the parameters eta and rho.
    """

    # just maiking sure no one tries to be funny...
    assert 0 <= kappa
    assert 0 <   a  <= 0.5
    assert 0 <= rho <= 1
    assert 1-rho > np.sinh((PI-2*a*PI)*kappa) / np.sinh(PI*kappa)

    posis = np.random.exponential(scale=1/kappa, size=N) % (2*PI)
    dists = geodesic_distances(posis)
    simis = overlap(dists, a)/(2*PI*a)
    probs = simis*slope(rho, kappa, a)

    return probs


def network_model(N, rho, kappa, a=0.5):
    """
    Network model as described in [1]. The output is the (empirical) positions
    of the nodes placed on the circle according to a von Mises distribution,
    followed by a tripple consisting of the (empirical) distribution of the
    pairwise distances, similarities and connection probabilities respectively.
    """
    probs = weighted_network_model(N = N,
                                   rho = rho,
                                   kappa = kappa,
                                   a = a)
    rands = np.random.uniform(size=round(N*(N-1)/2))
    A = squareform(np.where(probs>rands, 1, 0))
    G = nx.from_numpy_array(A)
    return G

# =============================================================================
#  ------------------------------ REFERENCES ---------------------------------
# =============================================================================

# [1] Not published yet.
