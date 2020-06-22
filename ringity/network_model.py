from ringity.distribution_functions import get_mu, mueta_to_alphabeta
from scipy.spatial.distance import squareform
from numpy import pi as PI

import scipy
import numpy as np
import networkx as nx

# =============================================================================
#  -------------------------------  PREPARATION -----------------------------
# =============================================================================

def overlap(dist, a):
    """
    Calculates the overlap of two boxes of length 2*pi*a on the circle for a
    given distance dist (measured from center to center).
    """
    x1 = np.where(2*PI*a-dist     > 0, 2*PI*a-dist    , 0)
    # for large box sizes (a>0) there is a second overlap
    x2 = np.where(dist-2*PI*(1-a) > 0, dist-2*PI*(1-a), 0)
    return x1 + x2


def connection_function(rho, eta, a, kappa):
    mu = get_mu(rho=rho, eta=eta, kappa=kappa, a=a)
    if   eta == 0:
        return lambda t : np.where(t < mu, 0., 1.)
    elif eta == 1:
        return lambda t : rho*np.ones_like(t)
    else:
        alpha, beta = mueta_to_alphabeta(mu, eta)
        return lambda t : scipy.stats.beta.cdf(t, a=alpha, b=beta)

# =============================================================================
#  ------------------------------  NETWORK MODEL ----------------------------
# =============================================================================

def network_model_samples(N, rho, eta=0, a=0.5, kappa=0):
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
    assert 0 <= eta <= 1
    assert 0 <= rho <= 1

    H = connection_function(rho=rho, eta=eta, kappa=kappa, a=a)

    positions = np.random.vonmises(mu=0, kappa=kappa, size=N)

    abs_dist = scipy.spatial.distance.pdist(positions.reshape(N,1))
    sqNDM = np.where(abs_dist<PI, abs_dist, 2*PI-abs_dist)
    sqNSM = overlap(sqNDM, a)/(2*PI*a)
    sqNPM = H(sqNSM)

    return (positions,) + tuple(map(squareform, [sqNDM, sqNSM, sqNPM]))


def network_model(N, rho, eta=0, kappa=0, a=0.5):
    """
    Network model as described in [1]. The output is the (empirical) positions
    of the nodes placed on the circle according to a von Mises distribution,
    followed by a tripple consisting of the (empirical) distribution of the
    pairwise distances, similarities and connection probabilities respectively.
    """
    positions, NDM, NSM, NPM = network_model_samples(N = N,
                                                           rho = rho,
                                                           eta = eta,
                                                           kappa = kappa,
                                                           a = a)
    R = squareform(np.random.uniform(size=round(N*(N-1)/2)))
    A = np.where(NPM>R, 1, 0)
    G = nx.from_numpy_array(A)
    return G

# =============================================================================
#  ------------------------------ REFERENCES ---------------------------------
# =============================================================================
"""
[1] Not published yet.
"""
