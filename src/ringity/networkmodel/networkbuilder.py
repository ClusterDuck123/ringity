from scipy.spatial.distance import squareform
from ringity.networkmodel.transformations import (
                                    string_to_distribution,
                                    string_to_similarity_function,
                                    string_to_probability_function
                                    )

import numpy as np
import scipy.stats as ss

from ringity.networkmodel.modelparameterbuilder import ModelParameterBuilder
from ringity.networkmodel.transformations import pw_angular_distance

"""This module describes the NetworkBuilder class.


We follow the "Builder desing principle".
Builder methods are sometimes called
 - "build", as in `build_model`
 - "infer", as in `infer_missing_parameters`
 - "instantiate", as in`instantiate_positions`
"""

# =============================================================================
#  ---------------------------- NETWORK BUILDER ------------------------------
# =============================================================================

class NetworkBuilder:
    """General class to build a network model.

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
        self.model_parameters = ModelParameterBuilder()

# ------------------------------------------------------------------
# --------------------------- PROPERTIES ---------------------------
# ------------------------------------------------------------------

# ----------------------- NETWORK PARAMETERS -----------------------
    @property
    def N(self):
        return self.model_parameters.size
    @property
    def response(self):
        return self.model_parameters.response
    @property
    def rate(self):
        return self.model_parameters.rate
    @property
    def delay(self):
        return self.model_parameters.delay
    @property
    def coupling(self):
        return self.model_parameters.coupling
    @property
    def density(self):
        return self.model_parameters.density

    @property
    def pwN(self):
        return (self.N * (self.N - 1)) // 2

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if hasattr(self, 'model'):
            raise AttributeError(f"Trying to set conflicting models "
                                 f"{value} != {self.model}")
        self._model = value

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

        if value.shape == (self.pwN, ):
            self._distances = value
        elif value.shape == (self.N, self.N):
            self._distances = squareform(value, checks = False)
        else:
            raise ValueError(f"Trying to set invalid distances!"
                             f"{value.shape, self.pwN}")


    @property
    def similarities(self):
        return squareform(self._similarities)

    @similarities.setter
    def similarities(self, value):
        if value.shape == (self.pwN, ):
            self._similarities = value
        elif value.shape == (self.N, self.N):
            self._similarities = squareform(value, checks = False)
        else:
            raise ValueError(f"Trying to set invalid similarities! {value.shape}")

    @property
    def probabilities(self):
        return squareform(self._probabilities)

    @probabilities.setter
    def probabilities(self, value):
        if value.shape == (self.pwN, ):
            self._probabilities = value
        elif value.shape == (self.N, self.N):
            self._probabilities = squareform(value, checks = False)
        else:
            raise ValueError(f"Trying to set invalid probabilities! {value.shape}")

# ------------------------------------------------------------------
#  --------------------------- METHODS ----------------------------
# ------------------------------------------------------------------
    def build_model(self):
        if self.rate == 0:
            if (self.response == 1) and (self.density != 1):
                self.model = "Uniform-ER (beta = 1, r = 1)"
                self._build_GRGG_model()
            if self.density == 1:
                self.model = "Uniform-Complete (beta = 1, rho = 1)"
                self._build_GRGG_model()
            if not hasattr(self, 'model'):
                self.model = "Uniform (beta = 1)"
                self._build_GRGG_model()

        elif self.rate > 200:
            self.model = "ER (beta = 0)"
            self._build_zero_distance_model()
        else:
            if self.coupling == 0:
                assert self.density == 0 # move to builder
                self.model = "Empty (c = 0)"
            if self.response == 1:
                self.model = "ER (r = 1)"
            if self.response == 0:
                assert self.density == 0 # move to builder
                self.model = "Empty (r = 0)"
            if self.density == 1:
                self.model = "Complete (rho = 1)"
            if not hasattr(self, 'model'):
                self.model = "General"
            
            self._build_general_model()
            

    # THINK ABOUT MOVING THESE METHODS OUTSIDE OF THE CLASS
    def _build_general_model(self):
        assert self.rate > 0
        self.set_distribution(
                            distn_arg = 'exponential',
                            scale = 1/self.rate)
        self.instantiate_positions()
        self.calculate_distances(
                            metric = 'euclidean',
                            circular = True)
        self.calculate_similarities(
                            r = self.response,
                            sim_func = 'box_cosine')
        self.calculate_probabilities(
                            prob_func = 'linear',
                            slope = self.coupling,
                            intercept = 0)

    def _build_zero_distance_model(self):
        # SET DELTA DISTRIBUTION
        # self.set_distribution(
        #                     distn_arg = 'exponential',
        #                     scale = scale)

        # self.instantiate_positions(self.N)
        if not np.isclose(self.density, self.coupling):
            raise ValueError(f"Conflicting values for zero-distance model found: "
                             f"density ({self.density}) != coupling ({self.coupling}).")
        self.positions = np.zeros(self.N)
        self.distances = squareform(np.zeros([self.N, self.N]))
        self.similarities = 1 - self.distances
        self.probabilities = self.density * self.similarities

    def _build_GRGG_model(self):
        assert self.rate == 0

        self.set_distribution(distn_arg = 'uniform', scale = 2*np.pi)
        self.instantiate_positions()
        self.calculate_distances(
                            metric = 'euclidean',
                            circular = True)
        self.calculate_similarities(
                            r = self.response,
                            sim_func = 'box_cosine')
        self.calculate_probabilities(
                            prob_func = 'linear',
                            slope = self.coupling,
                            intercept = 0)

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

        self.distribution = distn_arg

    def instantiate_positions(self, random_state = None):
        """Draws `N` elements on a circle from the distribution specified in
        self.distribution.


        `random_state` is an attribute that can be overwritten."""

        if random_state is None:
            random_state = self.__dict__.get('random_state', None)

        positions = self.distribution.rvs(self.N, random_state = random_state)
        self.positions = positions % (2*np.pi)


    def calculate_distances(self, circular = True, metric = 'euclidean'):
        self.distances = pw_angular_distance(self.positions, metric = metric)

    def calculate_similarities(self, sim_func = 'box_cosine', r = None):
        if r is not None:
            self.r = r

        if isinstance(sim_func, str):
            sim_func = string_to_similarity_function(sim_name = sim_func)
        elif not callable(sim_func):
            raise TypeError(f"Type of {sim_func} unknown.")

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
            raise TypeError(f"Type of {prob_func} unknown.")

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

        self.network = (self._probabilities >= coinflips).astype(int)



# =============================================================================
#  ------------------------------ REFERENCES ---------------------------------
# =============================================================================

# [1] Not published yet.
