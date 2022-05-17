from ringity.classes._distns import _get_rv
from ringity.classes.exceptions import ConflictingParametersError, ProvideParameterError, NotImplementedYetError
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import newton

import scipy
import numpy as np
import networkx as nx
import scipy.stats as ss

def network_model(N, 
                  a = 0.25, 
                  beta = None, 
                  rho = None, 
                  rate = None,
                  K = None, 
                  random_state = None,
                  return_positions = False,
                  verbose = False):
    
    rate = get_rate_parameter(rate = rate, beta = beta)
    scale = 1/rate if rate > 0 else np.inf
    K = get_interaction_parameter(K=K, rho=rho, rate=rate, a=a)
    
    if verbose:
        print(f"Rate parameter was calculated as:   lambda = {rate}")
        print(f"Interaction parameter was calculated as: K = {rate}")
    
    assert rate >= 0
    assert 0 <= K <= 1
    
    network_builder = NetworkBuilder(random_state = random_state)
    network_builder.set_distribution('exponential', scale = scale)
    network_builder.instantiate_positions(N)
    network_builder.calculate_distances(metric = 'euclidean', circular = True)
    network_builder.calculate_similarities(alpha = a, sim_func = 'box_cosine')
    network_builder.calculate_probabilities(prob_func = 'linear', slope = K, intercept = 0)
    network_builder.instantiate_network()
    
    G = nx.from_numpy_array(squareform(network_builder.network))
    
    if return_positions:
        return G, network_builder.positions
    else:
        return G

# =============================================================================
#  --------------------------- TRANSFORMATIONS ------------------------------
# =============================================================================
def box_cosine_similarity(dists, box_length):
    """Calculate the normalized overlap of two boxes on the circle."""    
    x1 = (box_length - dists).clip(0)
    
    # For box lengths larger than half of the circle there is a second overlap
    if box_length <= np.pi:
        x2 = 0
    else:
        x2 = (dists - (2*np.pi - box_length)).clip(0)
        
    total_overlap = x1 + x2
    return total_overlap / box_length    


def linear_probability(sims, slope, intercept):
    """Calculate linear transformation, capped at 0 an 1."""
    
    if slope < 0:
        Warning(f"Slope parameter is negative: {slope}")
    if intercept < 0:
        Warning(f"Intercept parameter is negative {intercept}")
        
    return (slope*sims + intercept).clip(0, 1)
    

def mean_similarity(rate, a):
    """Calculate expected mean similarity (equivalently maximal expected density). 
    
    This function assumies a (wrapped) exponential function and a cosine similarity 
    of box functions.
    """
    
    if np.isclose(rate, 0):
        return a
    # Python can't handle that properly; maybe the analytical expression can be simplified...
    if rate > 200:
        return 1.
    
    plamb = np.pi * rate
    
    # Alternatively:
    # numerator = 2*(np.sinh(plamb*(1-a)) * np.sinh(plamb*a))
    numerator = (np.cosh(plamb) - np.cosh(plamb*(1 - a*2)))
    denominator = (a*2*plamb * np.sinh(plamb))
    
    return 1 - numerator / denominator 
            
            
def density_to_interaction_strength(rho, a, rate = None, beta = None):
    rate = get_rate_parameter(rate = rate, beta = beta)
        
    mu_S = mean_similarity(rate = rate, a = a)
    if rho <= mu_S:
        return rho/mu_S
    else:
        raise ValueError("Please provide a lower density!")
        
def interaction_strength_to_density(K, a, rate = None, beta = None):
    rate = get_rate_parameter(rate = rate, beta = beta)
    rho = mean_similarity(a=a, rate = rate) * K
    return rho
        

            

# =============================================================================
#  ------------------------------- PARSERS ----------------------------------
# =============================================================================
def get_canonical_distn_name(distn_name):
    """Transform string to match a corresponding scipy.stats class."""
    distn_name = ''.join(filter(str.isalnum, distn_name.lower()))
    distn_name_converer = {
        'normal' : 'norm',
        'exponential' : 'expon',
        'wrappedexponential' : 'wrappedexpon'
    }
    return distn_name_converer.get(distn_name, distn_name)


def get_canonical_sim_name(sim_name):
    """Transform string to match name of a similarity function."""
    sim_name = sim_name.lower()
    if not sim_name.endswith('_similarity'):
        sim_name = sim_name + '_similarity'
    return sim_name


def get_canonical_prob_name(prob_name):
    """Transform string to match name of a probability function."""
    prob_name = prob_name.lower()
    if not prob_name.endswith('_probability'):
        prob_name = prob_name + '_probability'
    return prob_name


def string_to_distribution(distn_name,
                           **kwargs):
    """Take a string and returns a scipy.stats frozen random variable. 
    
    Parameters to the frozen random variable can be passed on via ``**kwargs`` 
    (e.g. loc = 5 or scale = 3). 
    The string is trying to match a corresponding scipy.stats class even 
    if there is no exact match. For example, the string 'exponential' will
    return the random variable scipy.stats.expon.
    See function ``get_canonical_distn_name`` for details on name conversion. 
    """
    
    # Check for edge cases
    if ('scale', np.inf) in kwargs.items():
        return ss.uniform(scale = 2*np.pi)
        
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
    
def get_rate_parameter(rate, beta):
    if beta is None and rate is None:
        raise ValueError(f"Please provide a distribution parameter: "
                         f"beta = {beta}, rate = {rate}")
    elif beta is not None and rate is not None:
        raise ValueError(f"Conflicting distribution parameters given: "
                         f"beta = {beta}, rate = {rate}")
    elif rate is not None:
        return rate
    elif beta is not None:
        if np.isclose(beta,0):
            return np.inf
        return np.tan(np.pi * (1-beta) / 2)      
    else:
        return ValueError("Unknown error. Please contact the developers "
                          "if you encounter this in the wild.")
                          
def get_interaction_parameter(K, rho, a, rate):
    if rho is None and K is None:
        raise ValueError(f"Please provide a distribution parameter: "
                         f"rho = {rho}, K = {K}")
    elif rho is not None and K is not None:
        raise ValueError(f"Conflicting distribution parameters given: "
                         f"rho = {rho}, K = {K}")
    elif K is not None:
        return K
    elif rho is not None:
        return density_to_interaction_strength(rho=rho, a=a, rate=rate)
    else:
        return ValueError("Unknown error. Please contact the developers "
                          "if you encounter this in the wild.")
        
    
# =============================================================================
#  ---------------------------- NETWORK BUILDER ------------------------------
# =============================================================================
    
    
class NetworkBuilder:
    """General use class to build a network model.
    
    Tailored to standard network model with parameters:
    - N ....... network parameter: network size.
    - rho ..... network parameter: expected network density.
    - beta .... distribution parameter: normalized rate parameter
    - alpha ... response parameter: length of the box functions.
    
    Model is not optimized for memory usage! In particular, distances, 
    similarities and probabilities are stored separately rather than 
    being transformed.
    """
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

# ------------------------------------------------------------------        
# --------------------------- PROPERTIES ---------------------------
# ------------------------------------------------------------------

# ----------------------- NETWORK PARAMETERS -----------------------
    @property
    def N(self):
        return self._N
    
    @N.setter
    def N(self, value):
        if hasattr(self, 'N') and self.N != value:
            raise AttributeError(f"Trying to set conflicting values "
                                 f"for N: {value} != {self.N}")
        self._N = value
        
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        if hasattr(self, 'alpha') and self.alpha != value:
            raise AttributeError(f"Trying to set conflicting values "
                                 f"for alpha: {value} != {self.alpha}")
        self._alpha = value
        
# -------------------------- NETWORK DATA --------------------------

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
# ---------------------------- METHODS ----------------------------
# ------------------------------------------------------------------
            
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
            
    def calculate_similarities(self, sim_func = 'box_cosine', alpha = None):
        if alpha is not None:
            self.alpha = alpha
            
        if isinstance(sim_func, str):
            sim_func = string_to_similarity_function(sim_name = sim_func)
        elif not callable(sim_func):
            raise TypeError(f"Type of {distn_arg} unknown.")
        
        self.similarities = sim_func(squareform(self.distances), 
                                     box_length = 2*np.pi*self.alpha)
        
    def calculate_probabilities(self, prob_func = 'linear', slope = None, intercept = None):
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