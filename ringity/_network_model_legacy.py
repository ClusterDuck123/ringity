from ringity.distribution_functions import mean_similarity, cdf_similarity, get_rate_parameter
from scipy.spatial.distance import pdist, squareform
from numpy import pi as PI

import scipy
import numpy as np
import networkx as nx

# =============================================================================
#  -------------------------------  PREPARATION -----------------------------
# =============================================================================

# TODO: collect parameters in a dictionary, e.g. network_parms

def get_positions(N, beta):
    if   beta == 0:
        return np.zeros(N)
    elif beta == 1:
        return np.random.uniform(0,2*PI, size=N)
    else:
        return np.random.exponential(scale=1/np.tan(PI*(1-beta)/2), size=N) % (2*PI)


def circular_distances(thetas):
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

def slope(rho, rate, a):
    mu_S = mean_similarity(rate,a)
    if rho <= mu_S:
        return rho/mu_S
    else:
        const = 1/np.sinh(PI*rate)
        def integral(k):
            term1 = np.sinh((1 + 2*a*(1/k-1))*PI*rate)
            term2 = (k*np.sinh((a*PI*rate)/k)*np.sinh(((a+k-2*a*k)*PI*rate)/k))/(a*PI*rate)
            return term1-term2
        return scipy.optimize.newton(
            func = lambda k: const*integral(k) + (1-cdf_similarity(1/k, rate, a)) - rho,
            x0 = rho/mu_S)

def get_a_min(rho, beta):
    if   beta == 0:
        return 0.
    elif beta == 1:
        return rho/2
    else:
        rate = np.tan(PI*(1-beta)/2)
        x = np.sinh(PI*rate)*(1-rho)
        return 1/2-np.log(np.sqrt(x**2+1)+x)/(2*PI*rate)


# =============================================================================
#  ------------------------------  NETWORK MODEL ----------------------------
# =============================================================================
def get_delays(N, param, parameter_type = 'delay'):
    if parameter_type == 'delay':
        return get_positions(N, param)
    else:
        assert False, "Not implemented yet!"

def delays_to_distances(dels):
    return circular_distances(dels)

def positions_to_distances(positions):
    return circular_distances(positions)

def distances_to_similarities(dists, a):
    return overlap(dists, a)/(2*PI*a)

def similarities_to_probabilities(simis, param, a, rho, parameter_type='rate'):
    rate = get_rate_parameter(param, parameter_type=parameter_type)
    rho_max = 1-np.sinh((PI-2*a*PI)*rate)/np.sinh(PI*rate)

    if np.isclose(rho,rho_max):
        # if rho is close to rho_max, all the similarities are 1.
        probs = np.sign(simis)
    elif rho < rho_max:
        k = slope(rho, rate, a)
        probs = (simis*k).clip(0,1)
    else:
        assert rho <= rho_max, "Please increase `a` or decrease `rho`!"

    return probs

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

    # just making sure no one tries to be funny...
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
        dists = circular_distances(posis)
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
        rate = np.tan(PI*(1-beta)/2)
        posis = np.random.exponential(scale=1/rate, size=N) % (2*PI)
        dists = circular_distances(posis)
        simis = overlap(dists, a)/(2*PI*a)

        rho_max = 1-np.sinh((PI-2*a*PI)*rate)/np.sinh(PI*rate)
        if np.isclose(rho,rho_max):
            probs = np.sign(simis)
        elif rho < 1-np.sinh((PI-2*a*PI)*rate)/np.sinh(PI*rate):
            k = slope(rho, rate, a)
            probs = (simis*k).clip(0,1)
        else:
            assert rho <= rho_max, "Please increase `a` or decrease `rho`!"

    if return_positions:
        return posis, probs
    else:
        return probs


def weighted_network_model2(N, rho, beta,
                            a = None,
                            posis = None,
                            return_positions =  False):
    """
    Returns samples of the Network model as described in [1]
    The outputs are samples of the
     - positions of the nodes placed on the circle according to a
       (wrapped) exponential distribution,
     - their pairwise distances
     - their similarities, given an 'activity window' of size a
     - their connection probabilities, given the expected density rho.
    """

    # STEP 0: calculate slope
    # STEP 1: Get posis
    # STEP 2: transform posis to distances
    # STEP 3: transform distances to similarities
    # STEP 4: transform similarities to probablities

    # just making sure no one tries to be funny...
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
        dists = circular_distances(posis)
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
        rate = np.tan(PI*(1-beta)/2)
        posis = np.random.exponential(scale=1/rate, size=N) % (2*PI)
        dists = circular_distances(posis)
        simis = overlap(dists, a)/(2*PI*a)

        rho_max = 1-np.sinh((PI-2*a*PI)*rate)/np.sinh(PI*rate)
        if np.isclose(rho,rho_max):
            probs = np.sign(simis)
        elif rho < 1-np.sinh((PI-2*a*PI)*rate)/np.sinh(PI*rate):
            k = slope(rho, rate, a)
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
#  --------------------------- OOP Here I come!! ------------------------------
# =============================================================================
class NetworkBuilder:
    pass

class GeneralNetworkBuilder:
    def __init__(self, N, rho, beta, a = None):
        self.N     = N
        self.rho   = rho
        self.beta  = beta
        self.rate  = np.tan(PI*(1-beta)/2)

        self.a_min = get_a_min(rho, beta)
        self.rho_max = 1-np.sinh((PI-2*self.a*PI)*self.rate)/np.sinh(PI*self.rate)

        if a is None:
            self.a = self.a_min
        else:
            self.a = a

        assert 0 <= self.beta <= 1
        assert 0 <= self.rho  <= 1

        assert self.a_min <= self.a <= 1

    def get_slope(self):
        mu_S = mean_similarity(self.rate, self.a)
        if self.rho <= mu_S:
            return self.rho/mu_S
        else:
            const = 1/np.sinh(PI*self.rate)
            def integral(k):
                term1 = np.sinh((1 + 2*self.a * (1/k-1))*PI*self.rate)
                term2 = (k*np.sinh((self.a * PI * self.rate)/k) *
                           np.sinh(((self.a + k - 2*k*self.a) * PI * self.rate)/k)) / \
                                    (self.a * PI * self.rate)
                return term1-term2
            self.slope = scipy.optimize.newton(
                func = lambda k: const*integral(k) +
                                 (1-cdf_similarity(1/k, self.rate, self.a)) -
                                 self.rho,
                x0 = self.rho/mu_S)

    def get_positions(self):
        self.positions = np.random.exponential(scale = 1/np.tan(PI*(1-self.beta)/2),
                                           size  = self.N) % (2*PI)
    def get_distances(self):
        self.distances = positions_to_distances(self.positions)

    def get_similarities(self):
        self.similarities = distances_to_similarities(self.distances, self.a)

    def get_probabilities(self):
        self.probabilities = (self.similarities * self.slope).clip(0,1)

    def generate_model(self, save_memory = False):
        self.get_slope()
        self.get_positions()

        self.get_distances()
        self.get_similarities()

        if save_memory:
            del self.distances

        self.get_probabilities()

        if save_memory:
            del self.similarities

    def get_network(self):
        coin_flip = np.random.uniform(size=int(N*(N-1)/2))
        A = squareform(np.where(self.probablities > coin_flip, 1, 0))
        return nx.from_numpy_array(A)

class ERNetworkBuilder:
    def __init__(self, N, rho):
        self.N     = N
        self.rho   = rho

    def generate_model(self):
        pass

    def get_network(self):
        return nx.erdos_renyi_graph(n = self.N, p = self.rho)


class WSNetworkBuilder:
    def __init__(self, N, rho, a = None):
        self.N     = N
        self.rho   = rho
        self.beta  = 1
        self.a_min = rho/2
        self.rho_max = 1 # Maybe....

        if a is None:
            self.a = self.a_min
        else:
            self.a = a

        assert 0 <= self.beta <= 1
        assert 0 <= self.rho  <= 1

        assert self.a_min <= self.a <= 1

    def get_slope(self):
        if np.isclose(self.a, self.a_min):
            self.slope = None
        elif self.rho <= self.a:
            self.slope = self.rho/self.a
        elif self.rho < 2*self.a:
            self.slope = self.a/(2*self.a - self.rho)
        else:
            assert self.rho <= 2*self.a, "Please increase `a` or decrease `rho`!"

    def get_positions(self):
        self.positions = np.random.exponential(scale = 1/np.tan(PI*(1-self.beta)/2),
                                               size  = self.N) % (2*PI)
    def get_distances(self):
        self.distances = positions_to_distances(self.positions)

    def get_similarities(self):
        self.similarities = distances_to_similarities(self.distances, self.a)

    def get_probabilities(self):
        if np.isclose(self.a, self.a_min):
            self.probabilities = np.sign(self.similarities)
        elif self.rho <= self.a:
            self.probabilities = (self.similarities).clip(0,1)
        elif self.rho < 2*self.a:
            self.probabilities = (self.similarities*self.slope).clip(0,1)
        else:
            assert False, "Something went wrong..."

    def generate_model(self, save_memory = False):
        self.get_slope()
        self.get_positions()

        self.get_distances()
        self.get_similarities()

        if save_memory:
            del self.distances

        self.get_probabilities()

        if save_memory:
            del self.similarities

    def get_network(self):
        coin_flip = np.random.uniform(size=int(N*(N-1)/2))
        A = squareform(np.where(self.probablities > coin_flip, 1, 0))
        return nx.from_numpy_array(A)


# =============================================================================
#  ------------------------------ REFERENCES ---------------------------------
# =============================================================================

# [1] Not published yet.
