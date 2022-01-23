from ringity.exceptions import DisconnectedGraphError

import os
import sys
import ringity
import unittest
import networkx as nx

class TestDiagramFunction(unittest.TestCase):
    def test_lipid_network(self):
        G = nx.read_edgelist("test_data/lipid_coexp_network.txt")
        dgm1 = ringity.core.diagram(G)
        dgm2 = ringity.classes.diagram.load_dgm("test_data/lipid_coexp_dgm.txt")
        for (pt1,pt2) in zip(dgm1, dgm2):
            self.assertAlmostEqual(pt1.birth,pt2.birth, places=5)
            self.assertAlmostEqual(pt1.death,pt2.death, places=5)
            
    def test_lipid_network_without_self_loops(self):
        G = nx.read_edgelist("test_data/lipid_coexp_network.txt")
        G = nx.convert_node_labels_to_integers(G)
        dgm1 = ringity.core.diagram(G)
        dgm2 = ringity.classes.diagram.load_dgm("test_data/lipid_coexp_dgm.txt")
        for (pt1,pt2) in zip(dgm1, dgm2):
            self.assertAlmostEqual(pt1.birth,pt2.birth, places=5)
            self.assertAlmostEqual(pt1.death,pt2.death, places=5)
            
    def test_lipid_network_with_integer_labels(self):
        G = nx.read_edgelist("test_data/lipid_coexp_network.txt")
        G.remove_edges_from(nx.selfloop_edges(G))
        dgm1 = ringity.core.diagram(G)
        dgm2 = ringity.classes.diagram.load_dgm("test_data/lipid_coexp_dgm.txt")
        for (pt1,pt2) in zip(dgm1, dgm2):
            self.assertAlmostEqual(pt1.birth,pt2.birth, places=5)
            self.assertAlmostEqual(pt1.death,pt2.death, places=5)

    def test_pathological_cases(self):
        pass

    def test_distance_matrix_flag_use_weight_compability(self):
        pass

    def test_distance_matrix_flag_vs_use_weight_with_zero_entries(self):
        pass

    def test_disconnection_error(self):
        G = nx.erdos_renyi_graph(10,0.5)
        G.add_node(11)
        self.assertRaises(DisconnectedGraphError, ringity.centralities.net_flow,G)

if __name__ == '__main__':
    unittest.main()
