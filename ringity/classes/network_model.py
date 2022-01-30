from ringity.distribution_functions import mean_similarity, cdf_similarity
from ringity.classes._distns import _get_rv
from ringity.classes.exceptions import ConflictingParametersError, ProvideParameterError, NotImplementedYetError
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import newton

import numpy as np
import networkx as nx
import scipy.stats as ss


# =============================================================================
#  --------------------------- OOP Here I come!! ------------------------------
# =============================================================================
def _convert_delay_to_rate(delay):
    scaled_delay = np.pi*delay/2
    return np.cos(scaled_delay) / np.sin(scaled_delay)

class PositionGenerator:
    def __init__(self, N,
                 distn = 'wrappedexpon',
                 random_state = None,
                 **kwargs):
        """
        The variable `distribution` can either be a string or a random variable
        form scipy.
        """
        self.N = N

        # In scipy-jargon a `rv` (random variable) is what we'd rather call a 
        # distribution *family* (e.g. a normal distribution). This corresponds
        # to our notion of `random_position`. 
        # A `frozen` random variable is what I would call an actual random variable
        # (e.g. a normal distribution with mu = 0, and sd = 1). This corresponds
        # to our notion of `frozen_position`.
        # Finally, a `rvs` (random variates) is an instantiation of a random variable
        # (e.g. 3.14159). This corresponds to our notion of `instantiated_position`.
        
        if   isinstance(distn, str):
            self.distribution_name = distn
            self.random_position = _get_rv(distn)
            self.frozen_position = self.random_position(**kwargs)
        elif isinstance(distn, ss._distn_infrastructure.rv_frozen):
            self.distribution_name = 'frozen'
            self.random_position = 'frozen'
            self.frozen_position = distn
        elif isinstance(distn, ss._distn_infrastructure.rv_generic):
            self.distribution_name = distn.name
            self.random_position = distn
            self.frozen_position = self.random_position(**kwargs)
        else:
            assert False, f"data type of distn recognized: ({type(self.distribution)})"
            
        self.current_position = self.frozen_position.rvs(size = N,
                                                         random_state = random_state,
                                                         **kwargs)
                                                       
    def redraw(self, random_state = None):
        self.current_position = self.frozen_position.rvs(size = N,
                                                         random_state = random_state)

class NetworkBuilder:
    def __init__(self, N,
                 distn = 'wrappedexpon',
                 rho = None,
                 k_max = None,
                 delay = None,
                 rate = None,
                 a = None,
                 **kwargs):
        self.N = N
        
        self._set_distribution_parameters(delay = delay, rate = rate)
        self._set_density_parameters(rho = rho, k_max = k_max)
        self._set_grid_parameters(a = a)
        
        # Check where this thing belongs
        self.rho_max = 1 - \
            np.sinh((np.pi - 2 * self.a * np.pi) * self.rate) / \
            np.sinh(np.pi * self.rate)

        assert 0 <= self.delay <= 1
        assert 0 <= self.rho <= 1

        assert self.a_min <= self.a <= 1
    
    def _set_density_parameters(self, rho = None, k_max = None):
        if rho is not None and k_max is not None:
            raise NotImplementedYetError("Conversion from ``rho`` to ``k_max`` not implemented yet!")
        elif rho:
            self.rho = rho
        elif k_max:
            raise NotImplementedYetError("Density parameter ``k_max`` not implemented yet!")
        else:
            raise ProvideParameterError("Please provide a density paramter!")
    
    def _set_distribution_parameters(self, delay = None, rate = None):
        # TODO: use Python's in-built getter and setter functions
        if delay is not None and rate is not None:
            
            delay_check = np.isclose(delay, np.tan(np.pi*(1-delay)/2))
            rate_check  = np.isclose(rate , 1 - 2/np.pi*np.arctan(self.rate))
            
            if not delay_check or not rate_check:
                raise ConflictingParametersError(f"Conflicting distribution parameters found!"
                                                  "rate = {rate}, delay = {delay}")
            else:
                self.delay = delay
                seld.rate  = rate
                
        elif delay:
            self.delay = delay
            self.rate = np.tan(np.pi*(1-delay)/2)
            
        elif rate:
            self.rate = rate
            self.delay = 1 - 2/np.pi*np.arctan(self.rate)
        else:
            raise ProvideParameterError("Please provide a distribution paramter!")
                        
    def _set_grid_parameters(self, a = None):
        self.a_min = self._set_a_min()
        
        if a is None:
            self.a = self.a_min
        else:
            self.a = a
        
            
    def _set_a_min(self):
        if np.isclose(self.delay, 0):
            self.a_min = 0.
        elif np.isclose(self.delay, 1):
            self.a_min = self.rho/2
        else:
            # TODO: break up the following expressions into interpretable terms
            x = np.sinh(np.pi*self.rate) * (1 - self.rho)
            return 1/2 - np.log(np.sqrt(x**2 + 1) + x)/(2*np.pi*self.rate)
        
    def get_positions(self, random_state = None):
        positions = PositionGenerator(N = N,
                                      distn = distn,
                                      random_state = random_state)
                                      
        self.positions = positions


class GeneralNetworkBuilder:
    def __init__(self, N, rho, delay, a=None):
        self.N = N
        self.rho = rho
        self.delay = delay
        self.rate = np.tan(np.pi * (1 - delay) / 2)

        self.a_min = get_a_min(rho, delay)
        self.rho_max = 1 - \
            np.sinh((np.pi - 2 * self.a * np.pi) * self.rate) / \
            np.sinh(np.pi * self.rate)

        if a is None:
            self.a = self.a_min
        else:
            self.a = a

        assert 0 <= self.delay <= 1
        assert 0 <= self.rho <= 1

        assert self.a_min <= self.a <= 1

    def get_slope(self):
        mu_S = mean_similarity(self.rate, self.a)
        if self.rho <= mu_S:
            return self.rho / mu_S
        else:
            const = 1 / np.sinh(np.pi * self.rate)

            def integral(k):
                term1 = np.sinh((1 + 2 * self.a * (1 / k - 1))
                                * np.pi * self.rate)
                term2 = (k * np.sinh((self.a * np.pi * self.rate) / k) *
                         np.sinh(((self.a + k - 2 * k * self.a) * np.pi * self.rate) / k)) / \
                    (self.a * np.pi * self.rate)
                return term1 - term2
            self.slope = newton(
                func=lambda k: const * integral(k) +
                (1 - cdf_similarity(1 / k, self.rate, self.a)) -
                self.rho,
                x0=self.rho / mu_S)

    def get_positions(self):
        self.positions = np.random.exponential(scale=1 / np.tan(np.pi * (1 - self.delay) / 2),
                                               size=self.N) % (2 * np.pi)

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
        self.delay = 1
        self.a_min = rho / 2
        self.rho_max = 1  # Maybe....

        if a is None:
            self.a = self.a_min
        else:
            self.a = a

        assert 0 <= self.delay <= 1
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
        self.positions = np.random.exponential(scale=1 / np.tan(np.pi * (1 - self.delay) / 2),
                                               size=self.N) % (2 * np.pi)

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
