import os
import unittest
import ringity as rng

from pathlib import Path
from ringity.ringscore.metric2ringscore import (
                            gap_ring_score,
                            geometric_ring_score,
                            linear_ring_score,
                            amplitude_ring_score,
                            entropy_ring_score)
from ringity.tda.pdiagram.generators import random_pdgm

DIRNAME_TMP = Path('test_data') / 'tmp'
FNAME_PDGM = DIRNAME_TMP / 'random_dgm.txt'

class TestReadAndWrite(unittest.TestCase):
    def test_save_and_load(self):
        pdgm1 = random_pdgm(2**5)
        rng.write_pdiagram(pdgm1, FNAME_PDGM)

        pdgm2 = rng.read_pdiagram(FNAME_PDGM)
        os.remove(FNAME_PDGM)
        
        self.assertEqual(pdgm1, pdgm2)

class TestRingScore(unittest.TestCase):
    def test_ring_score_flavours(self):
        pdgm = random_pdgm(2**5)
        pseq = pdgm.psequence(normalisation = 'signal')

        self.assertAlmostEqual(
                    gap_ring_score(pseq),
                    pdgm.ring_score(flavour = 'gap'))
        self.assertAlmostEqual(
                    linear_ring_score(pseq, nb_pers = 4),
                    pdgm.ring_score(flavour = 'linear', nb_pers = 4))
        self.assertAlmostEqual(
                    geometric_ring_score(pseq, nb_pers = 4, exponent = 3),
                    pdgm.ring_score(flavour = 'geometric', nb_pers = 4, exponent = 3))
        self.assertAlmostEqual(
                    amplitude_ring_score(pseq, nb_pers = 4),
                    pdgm.ring_score(flavour = 'amplitude', nb_pers = 4))
        self.assertAlmostEqual(
                    entropy_ring_score(pseq, nb_pers = 4),
                    pdgm.ring_score(flavour = 'entropy', nb_pers = 4))
        self.assertNotAlmostEqual(
                    geometric_ring_score(pseq, nb_pers = 4),
                    pdgm.ring_score(flavour = 'geometric', nb_pers = 4, exponent = 3))

if __name__ == '__main__':
    unittest.main()
