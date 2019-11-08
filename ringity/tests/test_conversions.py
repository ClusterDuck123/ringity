#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 19:05:19 2018

@author: myoussef
"""

import ringity
import unittest
import numpy as np
import networkx as nx

class TestConversions(unittest.TestCase):

    def test_ddict2dict2ddict_unweighted(self):
        E = nx.erdos_renyi_graph(100,0.17)
        d = dict(E.edges)
        dd = ringity.methods.dict2ddict(d)
        ddd = ringity.methods.ddict2dict(dd)
        dddd = ringity.methods.dict2ddict(ddd)
        self.assertEqual(dd, dddd)

    def test_ddict2dict2ddict_weighted(self):
        E = nx.erdos_renyi_graph(100,0.17)
        for (u, v) in E.edges():
            E[u][v]['weight'] = np.random.uniform(-1,1)

        d = dict(E.edges)
        dd = ringity.methods.dict2ddict(d)
        ddd = ringity.methods.ddict2dict(dd)
        dddd = ringity.methods.dict2ddict(ddd)
        self.assertEqual(dd, dddd)


if __name__ == '__main__':
    unittest.main()
