#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 19:05:19 2018

@author: myoussef
"""


import unittest
import numpy as np
import ringity as rng
import networkx as nx

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

        I = rng.current_flow_betweenness(G)
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

        I = rng.current_flow_betweenness(G)
        sample = set(np.round(I,3))

        self.assertEqual(sample, set([0.194, 0.267, 0.316, 0.321, 0.361, 0.417, 0.471]))


class TestNetFlowCentrality(unittest.TestCase):
    def setUp(self):
        self.G = nx.erdos_renyi_graph(50,0.3)

    def test_different_current_flow_matrix_calculations(self):
        A = nx.adjacency_matrix(self.G)
        C = rng.prepotential(self.G)
        B = rng.oriented_incidence_matrix(A).toarray()
        F2 = B@C
        F1 = rng.current_flow_matrix(A)
        equal = np.allclose(F1,F2)
        self.assertTrue(equal)

    def test_different_net_flow_calculations(self):
        bb1 = rng.stupid_current_distance(self.G)
        bb2 = rng.slow_current_distance(self.G)
        bb3 = rng.net_flow(self.G)

        self.assertEqual(set(bb1),set(bb2))
        self.assertEqual(set(bb2),set(bb3))
        self.assertEqual(set(bb3),set(self.G.edges))

        precision = 1e-10
        self.assertTrue(all([abs(bb1[key]-bb2[key]) < precision for key in bb1.keys()]))
        self.assertTrue(all([abs(bb2[key]-bb3[key]) < precision for key in bb1.keys()]))

    # def test_memory_speed_compability(self):
    #     edge_dict1 = rng.net_flow(self.G, efficiency = 'memory')
    #     edge_dict2 = rng.net_flow(self.G, efficiency = 'speed')
    # 
    #     np.testing.assert_allclose(np.array(list(edge_dict1.values())), 
    #                                np.array(list(edge_dict2.values())))


if __name__ == '__main__':
    unittest.main()
