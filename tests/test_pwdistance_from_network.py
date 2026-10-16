import numpy as np
import ringity as rty
import networkx as nx
import pytest

from collections import defaultdict


@pytest.fixture
def er_graph():
    G = nx.erdos_renyi_graph(2**5, p=0.25)
    return G


class TestPointCloud:
    def test_current_flow_centrality(self, er_graph):
        G = er_graph
        N = G.number_of_nodes()
        G_rng = G.copy()
        D_rng = rty.pwdistance_from_network(G_rng)

        bb = defaultdict(dict)
        for (u, v), value in nx.edge_current_flow_betweenness_centrality(G).items():
            bb[u][v] = {"net_flow": value}

        G_nx = nx.to_networkx_graph(bb)
        D_nx = nx.floyd_warshall_numpy(G_nx, nodelist=range(N), weight="net_flow")

        np.testing.assert_almost_equal(D_rng, D_nx * (N - 1) * (N - 2))
