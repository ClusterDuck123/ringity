from ringity.classes.network_model import NetworkBuilder

N = 2**5
rho = 0.1
delay = 0.123

network_builder = NetworkBuilder(N, delay=delay, rate=None, rho=rho, a=None, k_max=None)
network_builder.generate_positions()
network_builder.calculate_distances()
network_builder.calculate_similarities()
network_builder.calculate_probabilities()