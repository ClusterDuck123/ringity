from ringity.distribution_functions import mean_similarity, cdf_similarity, get_rate_parameter
from ringity.classes._distns import _get_rv
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import newton
from numpy import pi as PI

import numpy as np
import networkx as nx
import scipy.stats as ss


# =============================================================================
#  --------------------------- OOP Here I come!! ------------------------------
# =============================================================================
class PositionGenerator:
    def __init__(self, N,
                 distn='wrapped_exponential',
                 random_state=None,
                 **kwargs):
        """
        The variable `distribution` can either be a string or a random variable
        form scipy.
        """
        self.N = N

        if isinstance(distn, str):
            self.distribution = distn
            self.rv = _get_rv(distn)
            

        if isinstance(distn, ss._distn_infrastructure.rv_frozen):
            self.random_position = distn
        elif isinstance(distn, ss._distn_infrastructure.rv_generic):
            self.random_position = distn(**kwargs)
        else:
            assert False, f"data type of distn recognized: ({type(distn)})"

        self.fixed_position = self.random_position.rvs(size=N,
                                                       random_state=random_state)
                                                       
    def redraw(self, random_state=None):
        self.fixed_position = self.random_position.rvs(size=N,
                                                       random_state=random_state)

class NetworkBuilder:
    def __init__(self, N,
                 distn = 'wrapped_exponential',
                 rho = None,
                 beta = None,
                 rate = None,
                 a = None):
        self.N = N
        self.rho = _get_rho()
        self.beta = beta
        self.rate = np.tan(PI * (1 - beta) / 2)

        self.a_min = get_a_min(rho, beta)
        self.rho_max = 1 - \
            np.sinh((PI - 2 * self.a * PI) * self.rate) / \
            np.sinh(PI * self.rate)

        if a is None:
            self.a = self.a_min
        else:
            self.a = a

        assert 0 <= self.beta <= 1
        assert 0 <= self.rho <= 1

        assert self.a_min <= self.a <= 1
        
    def get_positions():
        pass


class GeneralNetworkBuilder:
    def __init__(self, N, rho, beta, a=None):
        self.N = N
        self.rho = rho
        self.beta = beta
        self.rate = np.tan(PI * (1 - beta) / 2)

        self.a_min = get_a_min(rho, beta)
        self.rho_max = 1 - \
            np.sinh((PI - 2 * self.a * PI) * self.rate) / \
            np.sinh(PI * self.rate)

        if a is None:
            self.a = self.a_min
        else:
            self.a = a

        assert 0 <= self.beta <= 1
        assert 0 <= self.rho <= 1

        assert self.a_min <= self.a <= 1

    def get_slope(self):
        mu_S = mean_similarity(self.rate, self.a)
        if self.rho <= mu_S:
            return self.rho / mu_S
        else:
            const = 1 / np.sinh(PI * self.rate)

            def integral(k):
                term1 = np.sinh((1 + 2 * self.a * (1 / k - 1))
                                * PI * self.rate)
                term2 = (k * np.sinh((self.a * PI * self.rate) / k) *
                         np.sinh(((self.a + k - 2 * k * self.a) * PI * self.rate) / k)) / \
                    (self.a * PI * self.rate)
                return term1 - term2
            self.slope = newton(
                func=lambda k: const * integral(k) +
                (1 - cdf_similarity(1 / k, self.rate, self.a)) -
                self.rho,
                x0=self.rho / mu_S)

    def get_positions(self):
        self.positions = np.random.exponential(scale=1 / np.tan(PI * (1 - self.beta) / 2),
                                               size=self.N) % (2 * PI)

    def get_distances(self):
        self.distances = positions_to_distances(self.positions)

    def get_similarities(self):
        self.similarities = distances_to_similarities(self.distances, self.a)

    def get_probabilities(self):
        self.probabilities = (self.similarities * self.slope).clip(0, 1)

    def generate_model(self, save_memory=False):
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
        coin_flip = np.random.uniform(size=int(N * (N - 1) / 2))
        A = squareform(np.where(self.probablities > coin_flip, 1, 0))
        return nx.from_numpy_array(A)


class ERNetworkBuilder:
    def __init__(self, N, rho):
        self.N = N
        self.rho = rho

    def generate_model(self):
        pass

    def get_network(self):
        return nx.erdos_renyi_graph(n=self.N, p=self.rho)


class WSNetworkBuilder:
    def __init__(self, N, rho, a=None):
        self.N = N
        self.rho = rho
        self.beta = 1
        self.a_min = rho / 2
        self.rho_max = 1  # Maybe....

        if a is None:
            self.a = self.a_min
        else:
            self.a = a

        assert 0 <= self.beta <= 1
        assert 0 <= self.rho <= 1

        assert self.a_min <= self.a <= 1

    def get_slope(self):
        if np.isclose(self.a, self.a_min):
            self.slope = None
        elif self.rho <= self.a:
            self.slope = self.rho / self.a
        elif self.rho < 2 * self.a:
            self.slope = self.a / (2 * self.a - self.rho)
        else:
            assert self.rho <= 2 * self.a, "Please increase `a` or decrease `rho`!"

    def get_positions(self):
        self.positions = np.random.exponential(scale=1 / np.tan(PI * (1 - self.beta) / 2),
                                               size=self.N) % (2 * PI)

    def get_distances(self):
        self.distances = positions_to_distances(self.positions)

    def get_similarities(self):
        self.similarities = distances_to_similarities(self.distances, self.a)

    def get_probabilities(self):
        if np.isclose(self.a, self.a_min):
            self.probabilities = np.sign(self.similarities)
        elif self.rho <= self.a:
            self.probabilities = (self.similarities).clip(0, 1)
        elif self.rho < 2 * self.a:
            self.probabilities = (self.similarities * self.slope).clip(0, 1)
        else:
            assert False, "Something went wrong..."

    def generate_model(self, save_memory=False):
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
        coin_flip = np.random.uniform(size=int(N * (N - 1) / 2))
        A = squareform(np.where(self.probablities > coin_flip, 1, 0))
        return nx.from_numpy_array(A)


# =============================================================================
#  ------------------------------ REFERENCES ---------------------------------
# =============================================================================

# [1] Not published yet.
