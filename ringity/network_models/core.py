import numpy as np
import networkx as nx

from scipy.spatial.distance import squareform
from ringity.network_models.networkbuilder import NetworkBuilder
from ringity.network_models.param_utils import (
                                get_response_parameter, 
                                get_rate_parameter, 
                                get_coupling_parameter
                                )

def network_model(N,
                  a = None,
                  r = None,
                  c = None,
                  alpha = None,
                  beta = None,
                  rho = None,
                  rate = None,
                  K = None,
                  random_state = None,
                  return_positions = False,
                  verbose = False):

    r = get_response_parameter(a=a, alpha=alpha, r=r)
    rate = get_rate_parameter(rate = rate, beta = beta)
    c = get_coupling_parameter(K=K, rho=rho, c=c, rate=rate, r=r)

    scale = 1/rate if rate > 0 else np.inf

    if verbose:
        print(f"Response parameter was calculated as: r = {r}")
        print(f"Rate parameter was calculated as:   lambda = {rate}")
        print(f"Coupling parameter was calculated as: c = {c}")

    assert rate >= 0
    assert 0 <= c <= 1

    network_builder = NetworkBuilder(random_state = random_state)
    network_builder.set_distribution('exponential',
                                     scale = scale)
    network_builder.instantiate_positions(N)
    network_builder.calculate_distances(metric = 'euclidean',
                                        circular = True)
    network_builder.calculate_similarities(r = r,
                                           sim_func = 'box_cosine')
    network_builder.calculate_probabilities(prob_func = 'linear',
                                            slope = c,
                                            intercept = 0)
    network_builder.instantiate_network()

    G = nx.from_numpy_array(squareform(network_builder.network))

    if return_positions:
        return G, network_builder.positions
    else:
        return G