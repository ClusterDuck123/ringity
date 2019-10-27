#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 19:05:19 2018

@author: myoussef
"""

import os
import sys

path = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.append(path)

import ringity
import unittest
import networkx as nx

RINGITY_PATH = os.path.dirname(ringity.__file__)

class TestDiagram(unittest.TestCase):

    def test_lipid_network(self):
        G = nx.read_edgelist(f"{RINGITY_PATH}/test_data/lipid_coexp_network.csv",
                            nodetype=int)
        G = nx.convert_node_labels_to_integers(G)
        dgm1 = ringity.core.diagram(G)
        dgm2 = ringity.classes.load_dgm(f"{RINGITY_PATH}/test_data/lipid_coexp_dgm.csv")
        for (pt1,pt2) in zip(dgm1, dgm2):
            self.assertAlmostEqual(pt1.birth,pt2.birth, places=3)
            self.assertAlmostEqual(pt1.death,pt2.death, places=3)


if __name__ == '__main__':
    unittest.main()
