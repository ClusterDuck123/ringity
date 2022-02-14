from ringity.classes.network_model import NetworkBuilder, PeriodicPositionGenerator
from ringity.classes._distns import get_frozen_wrappedexpon

N = 2**5
rho = 0.1
delay = 0.123

a = None
rate = None
k_max = None

def ringity_model(N, density_arg, distn_arg,
                  distn_argtype = 'delay',
                  density_argtype = 'density',
                  a = None, 
                  random_state = None):
    
    frozen_distn = get_frozen_wrappedexpon(arg = distn_arg, 
                                           argtype = distn_argtype)
    periodic_positions = PeriodicPositionGenerator(N, 
                                                   distn_arg = frozen_distn, 
                                                   random_state = random_state)
    
    network = NetworkBuilder(N, periodic_positions)
    network.generate_positions()
    network.calculate_distances()
    network.calculate_similarities()
    network.calculate_probabilities()
    network.generate_network()

    return network.network()
    
ringity_model(N=N, a=a, 
              density_arg = rho, density_argtype = 'density', 
              distn_arg = delay, distn_argtype = 'delay')