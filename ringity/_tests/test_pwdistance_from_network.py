import unittest
import numpy as np
import ringity as rng
import networkx as nx

from collections import defaultdict

class TestPointCloud(unittest.TestCase):
    def setUp(self):
        self.N = 2**5
        self.G = nx.erdos_renyi_graph(self.N, p = 0.25)

    def test_current_flow_centrality(self):
        G_rng = self.G.copy()
        D_rng = rng.pwdistance_from_network(G_rng)

        bb = defaultdict(dict)
        for (u, v), value in nx.edge_current_flow_betweenness_centrality(self.G).items():
            bb[u][v] = {'net_flow' : value}
            
        G_nx = nx.to_networkx_graph(bb)
        D_nx = nx.floyd_warshall_numpy(G_nx, nodelist = range(self.N), weight='net_flow')

        np.testing.assert_almost_equal(D_rng, D_nx * (self.N-1)*(self.N-2))

if __name__ == '__main__':
    unittest.main()
