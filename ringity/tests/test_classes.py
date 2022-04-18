import os
import sys
import random
import numpy as np

from ringity.generators.diagram import random_pdgm_point, random_pdgm

path = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.append(path)

import ringity
import unittest

RINGITY_PATH = os.path.dirname(ringity.__file__)

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
        self.signal_pt = ringity.classes.diagram.PersistenceDiagramPoint((0, self.number))
        self.noise_pt  = ringity.classes.diagram.PersistenceDiagramPoint((0, 0))

    def test_score1(self):
        dgm = ringity.classes.diagram.PersistenceDiagram(self.signal_pt for _ in range(100))
        self.assertAlmostEqual(dgm.ring_score(), 0, places = 8)

    def test_score2(self):
        dgm = ringity.classes.diagram.PersistenceDiagram(self.noise_pt for _ in range(100))
        dgm.append(self.signal_pt)
        self.assertAlmostEqual(dgm.ring_score(), 1, places=8)

    def test_birth_and_death_extractions(self):
        dgm = random_pdgm(100)
        self.assertEqual(dgm.births,tuple(pt.birth for pt in dgm))
        self.assertEqual(dgm.deaths,tuple(pt.death for pt in dgm))



if __name__ == '__main__':
    unittest.main()
