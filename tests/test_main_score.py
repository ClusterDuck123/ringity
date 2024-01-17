import random
import ringity
import unittest
import numpy as np

from pathlib import Path
from ringity.ringscore.metric2ringscore import ring_score_from_sequence

DIRNAME_TEST_DATA = Path('test_data') 
FNAME_PDGM = DIRNAME_TEST_DATA / 'lipid_coexp_dgm.txt'

class TestSyntheticExamples(unittest.TestCase):
    def setUp(self):
        self.signal = random.random()
        self.noise = [0]*random.randint(0,10)

    def test_score_of_zero1(self):
        multiple_signals = [self.signal]*50
        sequence = self.noise + multiple_signals + \
                   self.noise + multiple_signals + \
                   self.noise
        self.assertAlmostEqual(ring_score_from_sequence(sequence),
                               0, places=8)

    def test_score_of_zero2(self):
        sequence = ()
        self.assertAlmostEqual(ring_score_from_sequence(sequence),
                               0, places=8)

    def test_score_of_one(self):
        sequence = self.noise + [self.signal] + self.noise
        self.assertAlmostEqual(ring_score_from_sequence(sequence),
                               1, places=8)

    def test_score_of_half(self):
        sequence = self.noise + [self.signal] + \
                   self.noise + [self.signal] + \
                   self.noise
        self.assertAlmostEqual(ring_score_from_sequence(sequence),
                               0.5, places=8)

    def test_score_of_two_high_noises(self):
        noise_ratio2, noise_ratio1 = sorted(np.random.uniform(size=2))

        noise1 = self.signal*noise_ratio1
        noise2 = self.signal*noise_ratio2

        sequence = self.noise + [self.signal] + \
                   self.noise + [noise1] + \
                   self.noise + [noise2] + \
                   self.noise
        self.assertAlmostEqual(ring_score_from_sequence(sequence),
                               1-noise_ratio1/2 - noise_ratio2/4,
                               places = 8)

class TestKnownNetworks(unittest.TestCase):
    def test_lipid_network(self):
        dgm = ringity.read_pdiagram(FNAME_PDGM)
        self.assertAlmostEqual(dgm.ring_score(), 0.7669806588679011)

if __name__ == '__main__':
    unittest.main()
