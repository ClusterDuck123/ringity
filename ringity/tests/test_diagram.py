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
       
        
class TestDiagram(unittest.TestCase):
    
    def test_lipid_network(self):
        G = nx.read_edgelist('../test_data/lipid_coexp_network.csv', nodetype=int)
        dgm1 = ringity.core.diagram(G)
        dgm2 = ringity.classes.load_dgm('../test_data/lipid_coexp_dgm.csv')        
        self.assertEqual(dgm1, dgm2)
    
        
if __name__ == '__main__':
    unittest.main()
        