import random
import numpy as np

from ringity.tda.pdiagram.generators import random_pdgm_point, random_pdgm
from ringity.tda.pdiagram import pdiagram

import unittest

class TestDgmPt(unittest.TestCase):
    def setUp(self):
        self.number = random.random()
        self.dgm_pt = random_pdgm_point()

    def test_arithmetic_operations(self):
        np.testing.assert_allclose(np.array(self.dgm_pt+self.number),
                                   np.array(self.dgm_pt)+self.number)
        np.testing.assert_allclose(np.array(self.dgm_pt/self.number),
                                   np.array(self.dgm_pt)/self.number)

class TestDgm(unittest.TestCase):
    def setUp(self):
        self.number = random.random()
        self.signal_pt = pdiagram.PDiagramPoint((0, self.number))
        self.noise_pt  = pdiagram.PDiagramPoint((0, 0))

    def test_score1(self):
        dgm = pdiagram.PDiagram(self.signal_pt for _ in range(100))
        self.assertAlmostEqual(dgm.ring_score(), 0, places = 8)

    def test_score2(self):
        dgm = pdiagram.PDiagram(self.noise_pt for _ in range(100))
        dgm.append(self.signal_pt)
        self.assertAlmostEqual(dgm.ring_score(), 1, places = 8)

    def test_birth_and_death_extractions(self):
        dgm = random_pdgm(100)
        self.assertEqual(tuple(dgm.births), tuple(pt.birth for pt in dgm))
        self.assertEqual(tuple(dgm.deaths), tuple(pt.death for pt in dgm))



if __name__ == '__main__':
    unittest.main()
