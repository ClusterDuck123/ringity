import unittest
import numpy as np
import networkx as nx

from ringity._legacy.conversions import dict2ddict, ddict2dict

class TestConversions(unittest.TestCase):

    def test_ddict2dict2ddict_unweighted(self):
        E = nx.erdos_renyi_graph(100,0.17)
        d = dict(E.edges)
        dd = dict2ddict(d)
        ddd = ddict2dict(dd)
        dddd = dict2ddict(ddd)
        self.assertEqual(dd, dddd)

    def test_ddict2dict2ddict_weighted(self):
        E = nx.erdos_renyi_graph(100,0.17)
        for (u, v) in E.edges():
            E[u][v]['weight'] = np.random.uniform(-1,1)

        d = dict(E.edges)
        dd = dict2ddict(d)
        ddd = ddict2dict(dd)
        dddd = dict2ddict(ddd)
        self.assertEqual(dd, dddd)


if __name__ == '__main__':
    unittest.main()
