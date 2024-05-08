import networkx as nx

from scipy.spatial.distance import squareform
from ringity.networkmodel.networkbuilder import NetworkBuilder
from ringity.networkmodel.param_utils import parse_canonical_parameters

def network_model(N,
                  response = None,
                  coupling = None,
                  density = None,
                  rate = None,
                  beta = None,
                  random_state = None,
                  return_positions = False,
                  return_builder = False,
                  verbose = False,
                  a = None,
                  r = None,
                  c = None,
                  alpha = None,
                  rho = None,
                  K = None,
                  ):
    """Constructs and instantiates a network according to the network model.

    Can be seen as the "Director" of the builder class `NetworkBuilder`.
    """

    network_builder = NetworkBuilder(random_state = random_state)
    

    set_model_parameters(network_builder.model_parameters, N=N,
                         rate=rate, beta = beta,
                         response = response, r = r, a =  a, alpha = alpha,
                         coupling = coupling, c = c, K = K,
                         density = density, rho = rho,
                         verbose = verbose
                         )

    network_builder.build_model()

    if verbose:
        print(f"{network_builder.model} model detected.")

    network_builder.instantiate_network()

    G = nx.from_numpy_array(squareform(network_builder.network))

    output = [G]

    if return_positions:
        output.append(network_builder.positions)
    if return_builder:
        output.append(network_builder)

    if len(output) == 1:
        return output[0]
    else:
        return output
    

def set_model_parameters(model_parameters, N,
                         rate, beta,
                         response, r, a, alpha,
                         coupling, c, K,
                         density, rho,
                         verbose
                         ):
    params = {
        'N': N,
        'rate': rate, 'beta': beta,
        'response': response, 'r': r, 'a': a, 'alpha': alpha,
        'coupling': coupling, 'c': c, 'K': K,
        'density': density, 'rho': rho}

    model_params = parse_canonical_parameters(params)

    model_parameters.size = N
    model_parameters.rate = model_params['rate']
    model_parameters.response = model_params['response']
    model_parameters.coupling = model_params['coupling']
    model_parameters.density = model_params['density']

    # THIS NEEDS TO BE MOVED TO NETWORKBUILDER CLASS: E.G. set_parameters()
    # network_builder.set_parameters()

    model_parameters.infer_missing_parameters()

    if verbose:
        print(f"Response parameter was set  to: r = {model_parameters.response}")
        print(f"Rate parameter was set to:   rate = {model_parameters.rate}")
        print(f"Coupling parameter was set to:  c = {model_parameters.coupling}")
        print(f"Density parameter was set to: rho = {model_parameters.density}")
