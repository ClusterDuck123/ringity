import ringity as rng
import networkx as nx

N = 2**12
beta = 0.234
a = 0.234
rho = 0.123

G = rng.network_model(N=N, beta=beta, a=a, rho=rho)
print(nx.density(G))