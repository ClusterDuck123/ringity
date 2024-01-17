import inspect
import unittest
import numpy as np

from numpy.testing import assert_allclose
from ringity.ringscore.metric2ringscore import (
                            gap_ring_score,
                            geometric_ring_score,
                            linear_ring_score,
                            amplitude_ring_score,
                            entropy_ring_score)

ALL_RING_SCORES = {
        gap_ring_score,
        geometric_ring_score,
        linear_ring_score,
        amplitude_ring_score,
        entropy_ring_score
        }

INF_RING_SCORES = {
        geometric_ring_score,
        amplitude_ring_score,
        entropy_ring_score
        }
        
def _generate_message(seq, ring_score, rand_nb_pers, rand_base):
    """Informative error message."""
    nb_pers = min(10, len(seq))
    msg = f"\nseq: {seq[:nb_pers]} \n" \
          f"ring score: {ring_score.__name__} \n" \
          f"nb_pers: {rand_nb_pers} \n" \
          f"base: {rand_base} \n"
    return msg


class TestFiniteEdgeCases(unittest.TestCase):
    def setUp(self):
        self.rand_p0 = np.random.uniform(0, 10)
        self.rand_base = np.random.uniform(1, 10)
        self.rand_nb_pers = np.random.randint(2, 10)
        self.rand_zero_pads = np.random.randint(1, 10)
        self.rand_seq = np.random.uniform(size = self.rand_nb_pers)

    def test_empty_seq(self):
        seq = ()
        for ring_score in ALL_RING_SCORES:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = self.rand_nb_pers
            msg = _generate_message(seq, ring_score, self.rand_nb_pers, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 0., msg = msg)

    def test_noiseless_seq(self):
        seq = [self.rand_p0]
        for ring_score in ALL_RING_SCORES:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = self.rand_nb_pers
            msg = _generate_message(seq, ring_score, self.rand_nb_pers, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 1., msg = msg)

    def test_max_score_of_one(self):
        seq = [self.rand_p0] + [0]*self.rand_zero_pads
        for ring_score in ALL_RING_SCORES:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = self.rand_nb_pers
            msg = _generate_message(seq, ring_score, self.rand_nb_pers, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 1., msg = msg)

    def test_min_score_of_zero(self):
        seq = [self.rand_p0] * (self.rand_nb_pers + 1)
        for ring_score in ALL_RING_SCORES:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = self.rand_nb_pers
            msg = _generate_message(seq, ring_score, self.rand_nb_pers, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 0., msg = msg)

    def test_finiteness(self):
        for ring_score in ALL_RING_SCORES - {gap_ring_score}:
            msg = _generate_message(self.rand_seq,
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

            self.assertAlmostEqual(score1, score2, msg = msg)


class TestInfiniteCases(unittest.TestCase):
    def setUp(self):
        self.rand_p0 = np.random.uniform(0, 10)
        self.rand_base = np.random.uniform(1, 10)
        self.rand_zero_pads = np.random.randint(1, 10)

    def test_empty_seq(self):
        seq = ()
        for ring_score in INF_RING_SCORES:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = np.inf
            msg = _generate_message(seq, ring_score, np.inf, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 0., msg = msg)

    def test_delta_seq(self):
        seq = [self.rand_p0]
        for ring_score in INF_RING_SCORES:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = np.inf
            msg = _generate_message(seq, ring_score, np.inf, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 1., msg=msg)

    def test_max_score_of_one(self):
        seq = [self.rand_p0] + [0]*self.rand_zero_pads
        for ring_score in INF_RING_SCORES:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = np.inf
            msg = _generate_message(seq, ring_score, np.inf, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 1., msg=msg)

    def test_min_score_of_zero(self):
        delta = 1e-4
        nb_pers_geo = int(-np.log(delta) // np.log(self.rand_base))
        nb_pers_amp = int(1 / delta)
        nb_pers = max(nb_pers_geo, nb_pers_amp) + 1

        seq = [self.rand_p0] * nb_pers
        for ring_score in INF_RING_SCORES:
            kwargs = {}
            if 'base' in inspect.signature(ring_score).parameters:
                kwargs['base'] = self.rand_base
            if 'nb_pers' in inspect.signature(ring_score).parameters:
                kwargs['nb_pers'] = np.inf
            msg = _generate_message(seq, ring_score, np.inf, self.rand_base)
            self.assertAlmostEqual(ring_score(seq, **kwargs), 0., 
                                msg = msg, 
                                delta = delta)

    def test_infiniteness(self):
        seq = np.random.uniform(size = 5)
        for ring_score in INF_RING_SCORES - {entropy_ring_score}:
            msg = _generate_message(seq,
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

            self.assertAlmostEqual(score1, score2, msg = msg)

class TestTwoPersistenceIdentities(unittest.TestCase):
    def setUp(self):
        pseq_list = [np.random.uniform(size = N)
                                for _ in range(2**2)
                                    for N in range(2, 2**2)]

        self.p0_list = [sorted(pseq)[-1] for pseq in pseq_list]
        self.p1_list = [sorted(pseq)[-2] for pseq in pseq_list]

        self.gap_scores = [gap_ring_score(pseq) for pseq in pseq_list]
        self.amp_scores = [amplitude_ring_score(pseq, nb_pers = 2) for pseq in pseq_list]
        self.ent_scores = [entropy_ring_score(pseq, nb_pers = 2) for pseq in pseq_list]

    def test_gap_identities(self):
        test_scores1 = [1-p1/p0 for (p0,p1) in zip(self.p0_list, self.p1_list)]

        self.assertIsNone(assert_allclose(self.gap_scores, test_scores1))


    def test_amp_identities(self):
        test_scores1 = [(p0-p1)/(p0+p1) for (p0,p1) in zip(self.p0_list, self.p1_list)]
        test_scores2 = [gap/(2-gap) for gap in self.gap_scores]

        self.assertIsNone(assert_allclose(self.amp_scores, test_scores1))
        self.assertIsNone(assert_allclose(self.amp_scores, test_scores2))

    def test_ent_identities(self):
        def p1_over_p0_identity(p):
            numer = (p+1)*np.log(p+1) - p*np.log(p)
            denom = (p+1)*np.log(2)
            return 1 - numer/denom

        def amp_identity(x):
            numer = (2*x*np.arctanh(x) + np.log(1 - x**2))
            denom = np.log(4)
            return numer / denom

        test_scores1 = [p1_over_p0_identity(p0/p1)
                            for (p0,p1) in zip(self.p0_list, self.p1_list)]
        test_scores2 = [amp_identity(amp) for amp in self.amp_scores]

        self.assertIsNone(assert_allclose(self.ent_scores, test_scores1))
        self.assertIsNone(assert_allclose(self.ent_scores, test_scores2))


if __name__ == '__main__':
    unittest.main()
