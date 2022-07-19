from scipy.spatial.distance import pdist, squareform
from ringity.network_models.transformations import (
                                    string_to_distribution,
                                    string_to_similarity_function,
                                    string_to_probability_function
                                    )

import numpy as np
import scipy.stats as ss

from ringity.network_models.defaults import ARITHMETIC_PRECISION

# =============================================================================
#  ---------------------------- NETWORK BUILDER ------------------------------
# =============================================================================

class NetworkBuilder:
    """General use class to build a network model.

    Tailored to standard network model with parameters:
    - N ...... network parameter: network size.
    - rho .... network parameter: expected network density.
    - beta ... distribution parameter: normalized rate parameter
    - r ...... response parameter: length of the box functions.

    Model is not optimized for memory usage! In particular, distances,
    similarities and probabilities are stored separately rather than
    being transformed.
    """
    def __init__(self, **kwargs):
        self.__dict__ = kwargs
        self._N = None
        self._rate = None
        self._response = None
        self._coupling = None
        self._density = None

# ------------------------------------------------------------------
# --------------------------- PROPERTIES ---------------------------
# ------------------------------------------------------------------

# ----------------------- NETWORK PARAMETERS -----------------------
    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, value):
        if self._N and self.N != value:
            raise AttributeError(f"Trying to set conflicting values "
                                 f"for N: {value} != {self.N}")
        self._N = value

    @property
    def response(self):
        return self._response

    @response.setter
    def response(self, value):
        if self._response and self.response != value:
            raise AttributeError(f"Trying to set conflicting values "
                                 f"for response: {value} != {self.response}")
        self._response = value

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, value):
        if self._rate and self.rate != value:
            raise AttributeError(f"Trying to set conflicting values "
                                 f"for rate: {value} != {self.rate}")
        self._rate = value

    @property
    def coupling(self):
        return self._coupling

    @coupling.setter
    def coupling(self, value):
        if self._coupling and self.coupling != value:
            raise AttributeError(f"Trying to set conflicting values "
                                 f"for coupling: {value} != {self.coupling}")
        self._coupling = value

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, value):
        if hasattr(self, 'density') and self.density != value:
            raise AttributeError(f"Trying to set conflicting values "
                                 f"for density: {value} != {self.density}")
        self._density = value

# -------------------------- NETWORK META DATA --------------------------

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, value):
        assert isinstance(value, ss._distn_infrastructure.rv_frozen)
        self._distribution = value

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, value):
        assert np.all((0 <= value) & (value <= 2*np.pi))
        self._positions = value

    @property
    def distances(self):
        return squareform(self._distances)

    @distances.setter
    def distances(self, value):
        assert len(value.shape) == 1
        self._distances = value

    @property
    def similarities(self):
        return squareform(self._similarities)

    @similarities.setter
    def similarities(self, value):
        assert len(value.shape) == 1
        self._similarities = value

    @property
    def probabilities(self):
        return squareform(self._probabilities)

    @probabilities.setter
    def probabilities(self, value):
        assert len(value.shape) == 1
        self._probabilities = value

# ------------------------------------------------------------------
#  --------------------------- METHODS ----------------------------
# ------------------------------------------------------------------
    def infer_parameters(self):
        """Completes the set of network paramters, if possible.
        
        Besides of the network size (e.g. number of nodes), each 
        network model consistes of the following set of parameters:
        - distribution parameter (e.g. rate)
        - response window size (e.g. r)
        - coupling strength (e.g. c)
        - density (e.g. rho)
        
        However, a network model does only require 3 degrees of freedom, 
        so the abovementioned set is redundant. This function completes 
        the missing parameter if it is missing, or checks if the given 
        set is consistent with the model constraints.
        """
        
        params = [self.rate, self.response, self.coupling, self.density]
        
        if all(params):
            CHECK_PARAMS_CONSISTENCY
        elif sum(params) < 3:
            raise AttributeError("Not enough parameters given to infer "
                                 "redundant ones.")
        else:
            if not self.rate:
                pass
            if not self.response:
                pass
            if not self.coupling:
                pass
            if not self.density:
                pass
        
    def set_distribution(self,
                         distn_arg,
                         **kwargs):
        """Takes either a frozen distribution from scipy.stats or a string
        specifying a parametrized family of distributions and sets the property
        `frozen_distribution` to the corresponding frozen distribution. `**kwargs`
        will be used to specify the parameters of the distribution.
        """

        # Check if string or scipy distribution
        if isinstance(distn_arg, str):
            distn_arg =  string_to_distribution(distn_name = distn_arg, **kwargs)
        elif not isinstance(distn_arg, ss._distn_infrastructure.rv_frozen):
            raise TypeError(f"Type of {distn_arg} unknown.")

        self.distribution = self.distribution = distn_arg

    def instantiate_positions(self, N, random_state = None):
        """Draws `N` elements on a circle from the distribution specified in
        self.distribution.


        `random_state` is an attribute that can be overwritten."""
        self.N = N

        if random_state is None:
            random_state = self.__dict__.get('random_state', None)

        positions = self.distribution.rvs(N, random_state = random_state)
        self.positions = positions % (2*np.pi)


    def calculate_distances(self, circular = True, metric = 'euclidean'):
        abs_dists = pdist(self.positions.reshape(-1,1), metric = metric)

        if not circular:
            self.distances = abs_dists
        else:
            self.distances = np.where(abs_dists < np.pi,
                                      abs_dists,
                                      2*np.pi - abs_dists)

    def calculate_similarities(self, sim_func = 'box_cosine', r = None):
        if r is not None:
            self.r = r

        if isinstance(sim_func, str):
            sim_func = string_to_similarity_function(sim_name = sim_func)
        elif not callable(sim_func):
            raise TypeError(f"Type of {distn_arg} unknown.")

        self.similarities = sim_func(squareform(self.distances),
                                     box_length = 2*np.pi*self.response)

    def calculate_probabilities(self,
                                prob_func = 'linear',
                                slope = None,
                                intercept = None):
        # TODO: create interaction probability class
        if isinstance(prob_func, str):
            prob_func = string_to_probability_function(prob_name = prob_func)
        elif not callable(prob_func):
            raise TypeError(f"Type of {distn_arg} unknown.")

        self.probabilities = prob_func(self._similarities,
                                       slope = slope,
                                       intercept = intercept)

    def instantiate_network(self, random_state = None):
        """Draws a network from the probabilities specified in
        self.probabilities.


        `random_state` is an attribute that can be overwritten."""

        if random_state is None:
            random_state = self.__dict__.get('random_state', None)

        if not hasattr(self, 'N'):
            self.N = len(self.positions)

        np_rng = np.random.RandomState(random_state)
        coinflips = np_rng.uniform(size = (self.N * (self.N - 1)) // 2)

        self.network = self._probabilities >= coinflips


# =============================================================================
#  ------------------------------ REFERENCES ---------------------------------
# =============================================================================

# [1] Not published yet.
