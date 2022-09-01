import numpy as np
import networkx as nx

from scipy.spatial.distance import squareform
from ringity.classes.networkbuilder import NetworkBuilder
from ringity.network_models.param_utils import parse_canonical_parameters

def network_model(N,
                  response = None,
                  coupling = None,
                  density = None,
                  rate = None,
                  beta = None,
                  random_state = None,
                  return_positions = False,
                  verbose = False,
                  a = None,
                  r = None,
                  c = None,
                  alpha = None,
                  rho = None,
                  K = None,
                  ):
    """Constructs and instantiates a network according to the network model.
    """

    network_builder = NetworkBuilder(random_state = random_state)

    network_builder.N = N
    
    params = {
        'rate': rate, 'beta': beta,
        'response': response, 'r': r, 'a': a, 'alpha': alpha,
        'coupling': coupling, 'c': c, 'K': K,
        'density': density, 'rho': rho}
        
    model_params = parse_canonical_parameters(params)
    
    # THIS NEEDS TO BE MOVED TO NETWORKBUILDER CLASS: E.G. set_parameters()
    # network_builder.set_parameters()
    network_builder.rate = model_params['rate']
    network_builder.response = model_params['response']
    network_builder.coupling = model_params['coupling']
    network_builder.density = model_params['density']
    
    network_builder.infer_missing_parameters()
    network_builder.set_model()

    if verbose:
        print(f"Response parameter was set to:  r = {network_builder.response}")
        print(f"Rate parameter was set to: rate = {network_builder.rate}")
        print(f"Coupling parameter was set to:  c = {network_builder.coupling}")
        print(f"Density parameter was set to: rho = {network_builder.density}")
        print(f"{network_builder.model} model detected.")
    
    assert network_builder.rate >= 0
    assert 0 <= network_builder.coupling <= 1
    
    network_builder.build_model()
    network_builder.instantiate_network()

    G = nx.from_numpy_array(squareform(network_builder.network))

    if return_positions:
        return G, network_builder.positions
    else:
        return G
