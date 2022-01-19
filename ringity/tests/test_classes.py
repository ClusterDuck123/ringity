import os
import sys
import random
import numpy as np

path = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.append(path)

import ringity
import unittest

RINGITY_PATH = os.path.dirname(ringity.__file__)

class TestDgmPt(unittest.TestCase):
    def setUp(self):
        self.number = random.random()
        self.dgm_pt = ringity.classes.diagram.random_DgmPt()

    def test_arithmetic_operations(self):
        np.testing.assert_allclose(np.array(self.dgm_pt+self.number),
                                   np.array(self.dgm_pt)+self.number)
        np.testing.assert_allclose(np.array(self.dgm_pt/self.number),
                                   np.array(self.dgm_pt)/self.number)

class TestDgm(unittest.TestCase):
    def setUp(self):
        self.number = random.random()
        self.signal = ringity.classes.diagram.DgmPt(0, self.number)
        self.noise  = ringity.classes.diagram.DgmPt(0, 0)

    def test_score1(self):
        dgm = ringity.classes.diagram.Dgm(self.signal for _ in range(100))
        self.assertAlmostEqual(dgm.score, 0,
                               places=8)

    def test_score2(self):
        dgm = ringity.classes.diagram.Dgm(self.noise for _ in range(100))
        dgm = dgm.add(self.signal)
        self.assertAlmostEqual(dgm.score, 1,
                               places=8)

    def test_birth_and_death_extractions(self):
        dgm = ringity.classes.diagram.random_Dgm(length=100)
        self.assertEqual(dgm.births,[pt.birth for pt in dgm])
        self.assertEqual(dgm.deaths,[pt.death for pt in dgm])



if __name__ == '__main__':
    unittest.main()
