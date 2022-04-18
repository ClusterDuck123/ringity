import inspect
import unittest
import numpy as np

from ringity.ring_scores import (
                            gap_ring_score,
                            geometric_ring_score,
                            linear_ring_score,
                            amplitude_ring_score,
                            entropy_ring_score)

def generate_message(seq, ring_score, rand_nb_pers, rand_base):
    nb_pers = min(10, len(seq))
    msg = f"\nseq: {seq[:nb_pers]} \n" \
          f"ring score: {ring_score.__name__} \n" \
          f"nb_pers: {rand_nb_pers} \n" \
          f"base: {rand_base} \n"
    return msg

all_ring_scores = {
        gap_ring_score,
        geometric_ring_score,
        linear_ring_score,
        amplitude_ring_score,
        entropy_ring_score
        }

inf_ring_scores = {
        geometric_ring_score,
        amplitude_ring_score,
        entropy_ring_score
        }

class TestFiniteEdgeCases(unittest.TestCase):
    def setUp(self):
        self.rand_p0 = np.random.uniform(0, 10)
        self.rand_base = np.random.uniform(1, 10)
        self.rand_nb_pers = np.random.randint(2, 10)
        self.rand_zero_pads = np.random.randint(1, 10)
        self.rand_seq = np.random.uniform(size=self.rand_nb_pers)

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

    def test_finiteness(self):
        for ring_score in all_ring_scores - {gap_ring_score}:
            msg = generate_message(self.rand_seq,
                                   ring_score,
                                   self.rand_nb_pers,
                                   self.rand_base)
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base

            score1 = ring_score(self.rand_seq,
                                nb_pers = self.rand_nb_pers,
                                **kwargs)
            score2 = ring_score(list(self.rand_seq) + [min(self.rand_seq)],
                                nb_pers = self.rand_nb_pers,
                                **kwargs)

            self.assertAlmostEqual(score1, score2, msg=msg)


class TestInfiniteCases(unittest.TestCase):
    def setUp(self):
        self.rand_p0 = np.random.uniform(0, 10)
        self.rand_base = np.random.uniform(1, 10)
        self.rand_zero_pads = np.random.randint(1, 10)

    def test_empty_seq(self):
        seq = ()
        for ring_score in inf_ring_scores:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = np.inf
            msg = generate_message(seq, ring_score, np.inf, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 0., msg=msg)

    def test_noiseless_seq(self):
        seq = [self.rand_p0]
        for ring_score in inf_ring_scores:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = np.inf
            msg = generate_message(seq, ring_score, np.inf, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 1., msg=msg)

    def test_max_score_of_one(self):
        seq = [self.rand_p0] + [0]*self.rand_zero_pads
        for ring_score in inf_ring_scores:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = np.inf
            msg = generate_message(seq, ring_score, np.inf, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 1., msg=msg)

    def test_min_score_of_zero(self):
        delta = 1e-4
        nb_pers_geo = int(-np.log(delta) // np.log(self.rand_base))
        nb_pers_amp = int(1 / delta)
        nb_pers = max(nb_pers_geo, nb_pers_amp) + 1

        seq = [self.rand_p0] * nb_pers
        for ring_score in inf_ring_scores:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = np.inf
            msg = generate_message(seq, ring_score, np.inf, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 0., msg=msg, delta=delta)

    def test_infiniteness(self):
        seq = np.random.uniform(size = 5)
        for ring_score in inf_ring_scores - {entropy_ring_score}:
            msg = generate_message(seq,
                                   ring_score,
                                   np.inf,
                                   self.rand_base)
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base

            score1 = ring_score(seq,
                                nb_pers = np.inf,
                                **kwargs)
            score2 = ring_score(list(seq) + [0]*self.rand_zero_pads,
                                nb_pers = np.inf,
                                **kwargs)

            self.assertAlmostEqual(score1, score2, msg=msg)

if __name__ == '__main__':
    unittest.main()
