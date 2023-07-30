import unittest
import numpy as np
import networkx as nx

import ringity as rng
import ringity._legacy.legacy_centralities as legacy
import ringity.networks.centralities as cents

class TestCurrentFlowCentrality(unittest.TestCase):
    def test_network_1(self):
        """Test implementation against reported values from network 1 in
        Newman's paper Fig.3."""
        G_left = nx.complete_graph(range(1,6))
        G_right = nx.complete_graph(range(-5,0))

        G = nx.complete_graph(1)
        G.add_edges_from(G_left.edges())
        G.add_edges_from(G_right.edges())
        G.add_edges_from([(-1,0),(0,1),(-1,1)])

        G = nx.convert_node_labels_to_integers(G)

        I = legacy.current_flow_betweenness(G)
        sample = set(np.round(I,3))

        self.assertEqual(sample, set([0.269, 0.333, 0.67]))

    def test_network_2(self):
        """Test implementation against reported values from network 2 in
        Newman's paper Fig.3."""
        G_up = nx.path_graph([1,'A','B','C',-1])
        G_down = nx.path_graph([5,'a','b','c',-5])
        G_left = nx.complete_graph(range(1,6))
        G_right = nx.complete_graph(range(-5,0))

        G = nx.complete_graph(1)
        G.add_edges_from(G_left.edges())
        G.add_edges_from(G_right.edges())
        G.add_edges_from(G_up.edges())
        G.add_edges_from(G_down.edges())
        G.add_edges_from([(0,'A'),(0,'c')])

        G = nx.convert_node_labels_to_integers(G)

        I = legacy.current_flow_betweenness(G)
        sample = set(np.round(I,3))

        self.assertEqual(sample, set([0.194, 0.267, 0.316, 0.321, 0.361, 0.417, 0.471]))


class TestNetFlowCentrality(unittest.TestCase):
    def setUp(self):
        self.G = nx.erdos_renyi_graph(2**5, 2**(-1))

    def test_different_current_flow_matrix_calculations(self):
        A = nx.adjacency_matrix(self.G)
        C = legacy.prepotential(self.G)
        B = legacy.oriented_incidence_matrix(A).toarray()
        F2 = B@C
        F1 = legacy.current_flow_matrix(A)
        equal = np.allclose(F1,F2)
        self.assertTrue(equal)

    def test_different_net_flow_calculations(self):
        bb1 = cents.current_flow(self.G)
        bb2 = nx.edge_current_flow_betweenness_centrality(self.G)

        self.assertEqual(len(bb1),len(bb2))

        truth_values = [np.isclose(value, bb1[(min(edge), max(edge))]) 
                                for edge, value in bb2.items()]
        self.assertTrue(all(truth_values))

if __name__ == '__main__':
    unittest.main()
