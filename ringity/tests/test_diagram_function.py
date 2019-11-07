#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 19:05:19 2018

@author: myoussef
"""

import os
import sys

import ringity
import unittest
import networkx as nx

RINGITY_PATH = os.path.dirname(ringity.__file__)

class TestDiagram(unittest.TestCase):

    def test_lipid_network(self):
        G = nx.read_edgelist(f"{RINGITY_PATH}/test_data/lipid_coexp_network.txt",
                            nodetype=int)
        G = nx.convert_node_labels_to_integers(G)
        dgm1 = ringity.core.diagram(G)
        dgm2 = ringity.classes.load_dgm(f"{RINGITY_PATH}/test_data/lipid_coexp_dgm.txt")
        for (pt1,pt2) in zip(dgm1, dgm2):
            self.assertAlmostEqual(pt1.birth,pt2.birth, places=5)
            self.assertAlmostEqual(pt1.death,pt2.death, places=5)


if __name__ == '__main__':
    unittest.main()
