import os
import unittest
import ringity as rng
import networkx as nx

from pathlib import Path
from ringity.networkmeasures.centralities import net_flow
from ringity.classes.exceptions import DisconnectedGraphError

DIRNAME_RINGITY = Path(os.path.dirname(rng.__file__))
DIRNAME_TEST_DATA = DIRNAME_RINGITY / '_tests' / 'test_data'

FNAME_PDGM = DIRNAME_TEST_DATA / 'lipid_coexp_dgm.txt'
FNAME_NETWORK = DIRNAME_TEST_DATA / 'lipid_coexp_network.txt'

"""THIS MODULE NEEDS TO BE UPDATED!!"""


class TestDiagramFunction(unittest.TestCase):
    def setUp(self):
        self.G = nx.read_edgelist(FNAME_NETWORK)
        self.pdgm = rng.read_pdiagram(FNAME_PDGM)
    
    def test_lipid_network(self):    
        pdgm = rng.pdiagram_from_network(self.G)
        for (pt1,pt2) in zip(pdgm, self.pdgm):
            self.assertAlmostEqual(pt1.birth,pt2.birth, places = 5)
            self.assertAlmostEqual(pt1.death,pt2.death, places = 5)
            
    def test_lipid_network_with_integer_labels(self):
        G = nx.convert_node_labels_to_integers(self.G)
        pdgm = rng.pdiagram_from_network(G)
        for (pt1,pt2) in zip(pdgm, self.pdgm):
            self.assertAlmostEqual(pt1.birth, pt2.birth, places = 5)
            self.assertAlmostEqual(pt1.death, pt2.death, places = 5)
            
    def test_lipid_network_without_self_loops(self):
        G = self.G.copy()
        G.remove_edges_from(nx.selfloop_edges(G))
        pdgm = rng.pdiagram_from_network(G)
        for (pt1,pt2) in zip(pdgm, self.pdgm):
            self.assertAlmostEqual(pt1.birth, pt2.birth, places = 5)
            self.assertAlmostEqual(pt1.death, pt2.death, places = 5)

    def test_pathological_cases(self):
        pass

    def test_zero_weight_compability(self):
        pass

    def test_disconnection_error(self):
        G = nx.erdos_renyi_graph(10, 0.5)
        G.add_node(11)
        self.assertRaises(DisconnectedGraphError, net_flow, G)

if __name__ == '__main__':
    unittest.main()