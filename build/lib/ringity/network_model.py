from ringity.distribution_functions import mean_similarity, cdf_similarity
from scipy.spatial.distance import pdist, squareform
from numpy import pi as PI

import scipy
import numpy as np
import networkx as nx

# =============================================================================
#  -------------------------------  PREPARATION -----------------------------
# =============================================================================
def get_positions(N, beta):
    if   beta == 0:
        return np.zeros(N)
    elif beta == 1:
        return np.random.uniform(0,2*PI, size=N)
    else:
        return np.random.exponential(scale=1/np.tan(PI*(1-beta)/2), size=N) % (2*PI)


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
    # for box sizes with a>0 there is a second overlap
    else:
        x2 = (dist-2*PI*(1-a)).clip(0)
        return x1 + x2

def slope(rho, kappa, a):
    mu_S = mean_similarity(kappa,a)
    if rho <= mu_S:
        return rho/mu_S
    else:
        const = 1/np.sinh(PI*kappa)
        def integral(k): # This can probably be further simplified
            term1 = np.sinh((1 + 2*a*(1/k-1))*PI*kappa)
            term2 = (k*np.sinh((a*PI*kappa)/k)*np.sinh(((a+k-2*a*k)*PI*kappa)/k))/(a*PI*kappa)
            return term1-term2
        return scipy.optimize.newton(
            func = lambda k: const*integral(k) + (1-cdf_similarity(1/k, kappa, a)) - rho,
            x0 = rho/mu_S)

def get_a_min(rho, beta):
    if   beta == 0:
        return 0.
    elif beta == 1:
        return rho/2
    else:
        kappa = np.tan(PI*(1-beta)/2)
        x = np.sinh(PI*kappa)*(1-rho)
        return 1/2-np.log(np.sqrt(x**2+1)+x)/(2*PI*kappa)


# =============================================================================
#  ------------------------------  NETWORK MODEL ----------------------------
# =============================================================================

def weighted_network_model(N, rho, beta, a=None, return_positions=False):
    """
    Returns samples of the Network model as described in [1]
    The outputs are samples of the
     - positions of the nodes placed on the circle according to a
       (wrapped) exponential distribution,
     - their pairwise distances
     - their similarities, given an 'activity window' of size a
     - their connection probabilities, given the expected density rho.
    """

    # just maiking sure no one tries to be funny...
    assert 0 <= beta <= 1
    assert 0 <= rho  <= 1

    a_min   = get_a_min(rho, beta)

    if a is None:
        a = a_min

    assert 0 <= a <= 1

    if beta == 0 or a == 1:
        posis = np.zeros(N)
        simis = np.ones(int(N*(N-1)/2))
        k = rho
        probs = (simis*k).clip(0,1)
    elif beta == 1:
        posis = np.random.uniform(0,2*PI, size=N)
        dists = geodesic_distances(posis)
        simis = overlap(dists, a)/(2*PI*a)

        if np.isclose(a,a_min):
            probs = np.sign(simis)
        elif rho <= a:
            k = rho/a
            probs = (simis*k).clip(0,1)
        elif rho < 2*a:
            k = a/(2*a-rho)
            probs = (simis*k).clip(0,1)
        else:
            assert rho <= 2*a, "Please increase `a` or decrease `rho`!"
    else:
        kappa = np.tan(PI*(1-beta)/2)
        posis = np.random.exponential(scale=1/kappa, size=N) % (2*PI)
        dists = geodesic_distances(posis)
        simis = overlap(dists, a)/(2*PI*a)

        rho_max = 1-np.sinh((PI-2*a*PI)*kappa)/np.sinh(PI*kappa)
        if np.isclose(rho,rho_max):
            probs = np.sign(simis)
        elif rho < 1-np.sinh((PI-2*a*PI)*kappa)/np.sinh(PI*kappa):
            k = slope(rho, kappa, a)
            probs = (simis*k).clip(0,1)
        else:
            assert rho <= rho_max, "Please increase `a` or decrease `rho`!"

    if return_positions:
        return posis, probs
    else:
        return probs


def network_model(N, rho, beta, a=0.5, return_positions=False):
    """
    Network model as described in [1]. The output is the (empirical) positions
    of the nodes placed on the circle according to a von Mises distribution,
    followed by a tripple consisting of the (empirical) distribution of the
    pairwise distances, similarities and connection probabilities respectively.
    """
    if return_positions:
        posis, probs = weighted_network_model(N = N,
                                              rho = rho,
                                              beta = beta,
                                              a = a,
                                              return_positions = True)
    else:
        probs = weighted_network_model(N = N,
                                       rho = rho,
                                       beta = beta,
                                       a = a,
                                       return_positions = False)

    rands = np.random.uniform(size=int(N*(N-1)/2))
    A = squareform(np.where(probs>rands, 1, 0))
    G = nx.from_numpy_array(A)

    if return_positions:
        return posis, G
    else:
        return G

# =============================================================================
#  ------------------------------ REFERENCES ---------------------------------
# =============================================================================

# [1] Not published yet.
