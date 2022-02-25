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
def circular_distance(pts, period = 2*np.pi):
    """Takes a vector of points ``pts`` on the interval [0, ``period``] 
    and returns the shortest circular distance within that period. """
    
    assert ((pts >= 0) & (pts <= period)).all()
    
    abs_dists = pdist(pts.reshape(-1,1))
    return np.where(abs_dists < period / 2, abs_dists, period - abs_dists)

def overlap(dist, a):
    """
    Calculates the overlap of two boxes of length 2*pi*a on the circle for a
    given distance dist (measured from center to center).
    """
    
    # TODO: substitute ``a`` with ``box_length``
    
    x1 = (2*np.pi*a - dist).clip(0)
    
    if a <= 0.5:
        return x1
    # for box sizes with a>0 there is a second overlap
    else:
        x2 = (dist - 2*np.pi*(1-a)).clip(0)
        return x1 + x2
            
def normalized_overlap(dists, a):
    return overlap(dists, a)/(2*np.pi*a)
    
def similarities_to_probabilities(simis, param, a, rho, parameter_type='rate'):
    # This needs some heavy refactoring...
    rate = get_rate_parameter(param, parameter_type=parameter_type)
    rho_max = 1-np.sinh((np.pi-2*a*np.pi)*rate)/np.sinh(np.pi*rate)

    if np.isclose(rho,rho_max):
        # if rho is close to rho_max, all the similarities are 1.
        probs = np.sign(simis)
    elif rho < rho_max:
        k = slope(rho, rate, a)
        probs = (simis*k).clip(0,1)
    else:
        assert rho <= rho_max, "Please increase `a` or decrease `rho`!"

    return probs

class PeriodicPositionGenerator:
    def __init__(self, N,
                 distn_arg = 'wrappedexpon',
                 random_state = None,
                 **kwargs):
        """
        The variable `distribution` can either be a string or a random variable
        form scipy.
        """
        self.random_state = random_state
        self._N = N
        # In scipy-jargon a `rv` (random variable) is what we'd rather call a 
        # distribution *family* (e.g. a normal distribution). This corresponds
        # to our notion of `random_distribution`. 
        # A `frozen` random variable is what I would call an actual random variable
        # (e.g. a normal distribution with mu = 0, and sd = 1). This corresponds
        # to our notion of `frozen_distribution`.
        # Finally, a `rvs` (random variates) is an instantiation of a random variable
        # (e.g. 3.14159). This corresponds to our notion of `positions`.
        
        if isinstance(distn_arg, str):
            self.distribution_name = distn_arg
            self.random_distribution, self.distribution_args = _get_rv(distn_arg, **kwargs)
            self.frozen_distribution = self.random_distribution(**self.distribution_args)
        elif isinstance(distn_arg, ss._distn_infrastructure.rv_frozen):
            self.distribution_name = 'frozen'
            self.random_distribution = 'frozen'
            self.frozen_distribution = distn_arg
        elif isinstance(distn_arg, ss._distn_infrastructure.rv_generic):
            self.distribution_name = distn_arg.name
            self.random_distribution = distn_arg
            self.frozen_distribution = self.random_distribution(**kwargs)
        else:
            assert False, f"data type of distn recognized: ({type(self.distribution)})"
            
        self.generate_positions(random_state = self.random_state) 
                                                    
    @property
    def N(self):
        return self._N
        
    @N.setter
    def N(self, value):
        self._N = value
        generate_positions(random_state = self.random_state)
        
    
    @property
    def positions(self):
        return self._positions
                                        
    @positions.setter                                                   
    def positions(self, values):
        self._positions = values % (2*np.pi)                                             

    def generate_positions(self, random_state = None):
        if random_state is None:
            ranodm_state = self.random_state
        self._positions = self.frozen_distribution.rvs(size = self.N,
                                                       random_state = random_state)

class NetworkBuilder:
    def __init__(self, N,
                 distn_arg = 'wrappedexpon',
                 rho = None,
                 k_max = None,
                 delay = None,
                 rate = None,
                 a = None,
                 random_state = None,
                 **kwargs):
        
        # These attributes can be changed in isolation
        self.random_state = random_state
        
        # These properties influence the value of other properties
        self._N = N
        self._distn_arg = distn_arg
        self._kwargs = kwargs
        self._set_distribution_parameters(delay = delay, rate = rate)
        self._set_density_parameters(rho = rho, k_max = k_max)
        self._set_grid_parameters(a = a)
        self._position_generator = PositionGenerator(N = self.N,
                                                     distn_arg = self.distn,
                                                     rate = self.rate,
                                                     delay = self.delay,
                                                     random_state = self.random_state,
                                                     **kwargs)
        
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
        
    def generate_positions(self, random_state = None):
        if random_state is None:
            random_state = self.random_state
        self._positions = self._position_generator.generate_positions(random_state)
        
    def calculate_distances(self):
        self.distances = circular_distance(self.positions)
    
    def calculate_similarities(self, calculate_from = 'distances'):
        self.similarities = normalized_overlap(self.distances, a = self.a)
    
    def calculate_probabilities(self, calculate_from = 'similarities'):
        pass
        
    def instantiate_network(self):
        pass
        
# =============================================================================
#  ---------------------------------- Legacy --------------------------------
# =============================================================================
def slope(rho, rate, a):
    mu_S = mean_similarity(rate,a)
    if rho <= mu_S:
        return rho/mu_S
    else:
        const = 1/np.sinh(np.pi*rate)
        def integral(k):
            term1 = np.sinh((1 + 2*a*(1/k-1))*np.pi*rate)
            term2 = (k*np.sinh((a*np.pi*rate)/k)*np.sinh(((a+k-2*a*k)*np.pi*rate)/k))/(a*np.pi*rate)
            return term1-term2
        return scipy.optimize.newton(
            func = lambda k: const*integral(k) + (1-cdf_similarity(1/k, rate, a)) - rho,
            x0 = rho/mu_S)
            
def get_rate_parameter(parameter, parameter_type):
    if   parameter_type.lower() == 'rate':
        return parameter
    elif parameter_type.lower() == 'shape':
        return 1/parameter
    elif parameter_type.lower() == 'delay':
        return np.cos(np.pi*parameter/2) / np.sin(np.pi*parameter/2)
    else:
        assert False, f"Parameter type '{parameter_type}' not known! " \
                       "Please choose between 'rate', 'shape' and 'delay'."

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
