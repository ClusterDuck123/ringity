from centralities     import net_flow as net_flow1
from new_centralities import net_flow as net_flow2

import time
import numpy as np
import networkx as nx

np.random.seed(123)

G = nx.erdos_renyi_graph(2**9, 0.05)
A1 = net_flow1(G)
A2 = net_flow2(G)

for e in A1.keys():
    assert np.isclose(A1[e],A2[e])

from memory_profiler import profile

@profile
def my_func():
    N = 2**12
    m = 5
    G = nx.barabasi_albert_graph(N,m, seed=12345)
    _ = net_flow2(G)

if __name__ == '__main__':
    t1 = time.time()
    my_func()
    t2 = time.time()
    print(t2-t1)
