import inspect
import unittest
import numpy as np
import ringity as rng

def generate_message(seq, ring_score, rand_nb_pers, rand_base):
    msg = f"seq: {seq} \n" \
          f"ring score: {ring_score.__name__} \n" \
          f"nb_pers: {rand_nb_pers} \n" \
          f"base: {rand_base} \n"
    return msg

all_ring_scores = {
        rng.gap_ring_score,
        rng.geometric_ring_score,
        rng.linear_ring_score,
        rng.amplitude_ring_score,
        rng.entropy_ring_score
        }

class TestFiniteEdgeCases(unittest.TestCase):
    def setUp(self):
        self.rand_p0 = np.random.uniform(0, 10)
        self.rand_base = np.random.uniform(0, 10)
        self.rand_nb_pers = np.random.randint(2, 10)
        self.rand_zero_pads = np.random.randint(1, 10)
        
    def test_empty_seq(self):
        seq = ()
        for ring_score in all_ring_scores:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = self.rand_nb_pers
            msg = generate_message(seq, ring_score, self.rand_nb_pers, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 0., msg=msg)
            
    def test_noiseless_seq(self):
        seq = [self.rand_p0]
        for ring_score in all_ring_scores:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = self.rand_nb_pers
            msg = generate_message(seq, ring_score, self.rand_nb_pers, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 1., msg=msg)

    def test_max_score_of_one(self):
        seq = [self.rand_p0] + [0]*self.rand_zero_pads
        for ring_score in all_ring_scores:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = self.rand_nb_pers
            msg = generate_message(seq, ring_score, self.rand_nb_pers, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 1., msg=msg)
            
    def test_min_score_of_zero(self):
        seq = [self.rand_p0] * (self.rand_nb_pers + 1)
        for ring_score in all_ring_scores:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = self.rand_nb_pers
            msg = generate_message(seq, ring_score, self.rand_nb_pers, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 0., msg=msg)
            
class TestInfiniteCases(unittest.TestCase):
    def setUp(self):
        self.rand_p0 = np.random.uniform(0, 10)
        self.rand_base = np.random.uniform(0, 10)
        self.rand_zero_pads = np.random.randint(1, 10)
        
    def test_empty_seq(self):
        seq = ()
        for ring_score in all_ring_scores:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = np.inf
            msg = generate_message(seq, ring_score, np.inf, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 0., msg=msg)
            
    def test_noiseless_seq(self):
        seq = [self.rand_p0]
        for ring_score in all_ring_scores:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = np.inf
            msg = generate_message(seq, ring_score, np.inf, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 1., msg=msg)

    def test_max_score_of_one(self):
        seq = [self.rand_p0] + [0]*self.rand_zero_pads
        for ring_score in all_ring_scores:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = np.inf
            msg = generate_message(seq, ring_score, np.inf, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 1., msg=msg)
            
#    def test_min_score_of_zero(self):
#        seq = [self.rand_p0] * (self.rand_nb_pers + 1)
#        for ring_score in all_ring_scores:
#            kwargs = {}
#            if 'base' in inspect.signature(ring_score).parameters:
#                kwargs['base'] = self.rand_base
#            if 'nb_pers' in inspect.signature(ring_score).parameters:
#                kwargs['nb_pers'] = self.rand_nb_pers
#            msg = generate_message(seq, ring_score, np.inf, self.rand_base)
#            self.assertAlmostEqual(ring_score(seq, **kwargs), 0., msg=msg)

if __name__ == '__main__':
    unittest.main()