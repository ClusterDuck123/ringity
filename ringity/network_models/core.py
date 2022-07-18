import numpy as np
import networkx as nx

from scipy.spatial.distance import squareform
from ringity.network_models.networkbuilder import NetworkBuilder
from ringity.network_models.param_utils import (
                                parse_response_parameter,
                                parse_rate_parameter,
                                parse_coupling_parameter
                                )

def network_model(N,
                  a = None,
                  r = None,
                  c = None,
                  response = None,
                  coupling = None,
                  alpha = None,
                  beta = None,
                  rho = None,
                  rate = None,
                  K = None,
                  random_state = None,
                  return_positions = False,
                  verbose = False):

    network_builder = NetworkBuilder(random_state = random_state)

    network_builder.N = N
    network_builder.rate = parse_rate_parameter(
                                    rate = rate,
                                    beta = beta
                                    )
    network_builder.response = parse_response_parameter(
                                    r = r,
                                    a = a,
                                    alpha = alpha,
                                    response = response
                                    )
    network_builder.coupling = parse_coupling_parameter(
                                    c = c,
                                    K = K,
                                    coupling = coupling
                                    )
    network_builder.rho = rho

    if verbose:
        print(f"Response parameter was calculated as:  r = {network_builder.response}")
        print(f"Rate parameter was calculated as: lambda = {network_builder.rate}")
        print(f"Coupling parameter was calculated as:  c = {network_builder.coupling}")
        print(f"Density parameter was calculated as: rho = {network_builder.rho}")
    
    assert network_builder.rate >= 0
    assert 0 <= network_builder.coupling <= 1
    
    scale = 1/network_builder.rate if network_builder.rate > 0 else np.inf

    network_builder.set_distribution('exponential',
                                     scale = scale)
    network_builder.instantiate_positions(N)
    network_builder.calculate_distances(metric = 'euclidean',
                                        circular = True)
    network_builder.calculate_similarities(r = network_builder.response,
                                           sim_func = 'box_cosine')
    network_builder.calculate_probabilities(prob_func = 'linear',
                                            slope = network_builder.coupling,
                                            intercept = 0)
    network_builder.instantiate_network()

    G = nx.from_numpy_array(squareform(network_builder.network))

    if return_positions:
        return G, network_builder.positions
    else:
        return G
