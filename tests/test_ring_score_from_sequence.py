import inspect
import pytest
import numpy as np

from numpy.testing import assert_allclose
from ringity.ringscore.metric2ringscore import (
    gap_ring_score,
    geometric_ring_score,
    linear_ring_score,
    amplitude_ring_score,
    entropy_ring_score,
)

ALL_RING_SCORES = {
    gap_ring_score,
    geometric_ring_score,
    linear_ring_score,
    amplitude_ring_score,
    entropy_ring_score,
}

INF_RING_SCORES = {geometric_ring_score, amplitude_ring_score, entropy_ring_score}


def _generate_message(seq, ring_score, rand_nb_pers, rand_base):
    """Informative error message."""
    nb_pers = min(10, len(seq))
    msg = (
        f"\nseq: {seq[:nb_pers]} \n"
        f"ring score: {ring_score.__name__} \n"
        f"nb_pers: {rand_nb_pers} \n"
        f"base: {rand_base} \n"
    )
    return msg


@pytest.fixture
def rand_p0():
    return np.random.uniform(0, 10)


@pytest.fixture
def rand_base():
    return np.random.uniform(1, 10)


@pytest.fixture
def rand_nb_pers():
    return np.random.randint(2, 10)


@pytest.fixture
def rand_zero_pads():
    return np.random.randint(1, 10)


@pytest.fixture
def rand_seq(rand_nb_pers):
    return np.random.uniform(size=rand_nb_pers)


class TestFiniteEdgeCases:
    def test_empty_seq(self, rand_base, rand_nb_pers):
        seq = ()
        for ring_score in ALL_RING_SCORES:
            kwargs = {}
            if "base" in inspect.signature(ring_score).parameters:
                kwargs["base"] = rand_base
            if "nb_pers" in inspect.signature(ring_score).parameters:
                kwargs["nb_pers"] = rand_nb_pers
            msg = _generate_message(seq, ring_score, rand_nb_pers, rand_base)
            assert ring_score(seq, **kwargs) == pytest.approx(0.0), msg

    def test_noiseless_seq(self, rand_p0, rand_base, rand_nb_pers):
        seq = [rand_p0]
        for ring_score in ALL_RING_SCORES:
            kwargs = {}
            if "base" in inspect.signature(ring_score).parameters:
                kwargs["base"] = rand_base
            if "nb_pers" in inspect.signature(ring_score).parameters:
                kwargs["nb_pers"] = rand_nb_pers
            msg = _generate_message(seq, ring_score, rand_nb_pers, rand_base)
            assert ring_score(seq, **kwargs) == pytest.approx(1.0), msg

    def test_max_score_of_one(self, rand_p0, rand_zero_pads, rand_base, rand_nb_pers):
        seq = [rand_p0] + [0] * rand_zero_pads
        for ring_score in ALL_RING_SCORES:
            kwargs = {}
            if "base" in inspect.signature(ring_score).parameters:
                kwargs["base"] = rand_base
            if "nb_pers" in inspect.signature(ring_score).parameters:
                kwargs["nb_pers"] = rand_nb_pers
            msg = _generate_message(seq, ring_score, rand_nb_pers, rand_base)
            assert ring_score(seq, **kwargs) == pytest.approx(1.0), msg

    def test_min_score_of_zero(self, rand_p0, rand_nb_pers, rand_base):
        seq = [rand_p0] * (rand_nb_pers + 1)
        for ring_score in ALL_RING_SCORES:
            kwargs = {}
            if "base" in inspect.signature(ring_score).parameters:
                kwargs["base"] = rand_base
            if "nb_pers" in inspect.signature(ring_score).parameters:
                kwargs["nb_pers"] = rand_nb_pers
            msg = _generate_message(seq, ring_score, rand_nb_pers, rand_base)
            assert ring_score(seq, **kwargs) == pytest.approx(0.0), msg

    def test_finiteness(self, rand_seq, rand_nb_pers, rand_base):
        for ring_score in ALL_RING_SCORES - {gap_ring_score}:
            msg = _generate_message(rand_seq, ring_score, rand_nb_pers, rand_base)
            kwargs = {}
            if "base" in inspect.signature(ring_score).parameters:
                kwargs["base"] = rand_base

            score1 = ring_score(rand_seq, nb_pers=rand_nb_pers, **kwargs)
            score2 = ring_score(
                list(rand_seq) + [min(rand_seq)],
                nb_pers=rand_nb_pers,
                **kwargs,
            )

            assert score1 == pytest.approx(score2), msg


class TestInfiniteCases:
    def test_empty_seq(self, rand_base):
        seq = ()
        for ring_score in INF_RING_SCORES:
            kwargs = {}
            if "base" in inspect.signature(ring_score).parameters:
                kwargs["base"] = rand_base
            if "nb_pers" in inspect.signature(ring_score).parameters:
                kwargs["nb_pers"] = np.inf
            msg = _generate_message(seq, ring_score, np.inf, rand_base)
            assert ring_score(seq, **kwargs) == pytest.approx(0.0), msg

    def test_delta_seq(self, rand_p0, rand_base):
        seq = [rand_p0]
        for ring_score in INF_RING_SCORES:
            kwargs = {}
            if "base" in inspect.signature(ring_score).parameters:
                kwargs["base"] = rand_base
            if "nb_pers" in inspect.signature(ring_score).parameters:
                kwargs["nb_pers"] = np.inf
            msg = _generate_message(seq, ring_score, np.inf, rand_base)
            assert ring_score(seq, **kwargs) == pytest.approx(1.0), msg

    def test_max_score_of_one(self, rand_p0, rand_zero_pads, rand_base):
        seq = [rand_p0] + [0] * rand_zero_pads
        for ring_score in INF_RING_SCORES:
            kwargs = {}
            if "base" in inspect.signature(ring_score).parameters:
                kwargs["base"] = rand_base
            if "nb_pers" in inspect.signature(ring_score).parameters:
                kwargs["nb_pers"] = np.inf
            msg = _generate_message(seq, ring_score, np.inf, rand_base)
            assert ring_score(seq, **kwargs) == pytest.approx(1.0), msg

    def test_min_score_of_zero(self, rand_base, rand_p0):
        delta = 1e-4
        nb_pers_geo = int(-np.log(delta) // np.log(rand_base))
        nb_pers_amp = int(1 / delta)
        nb_pers = max(nb_pers_geo, nb_pers_amp) + 1

        seq = [rand_p0] * nb_pers
        for ring_score in INF_RING_SCORES:
            kwargs = {}
            if "base" in inspect.signature(ring_score).parameters:
                kwargs["base"] = rand_base
            if "nb_pers" in inspect.signature(ring_score).parameters:
                kwargs["nb_pers"] = np.inf
            msg = _generate_message(seq, ring_score, np.inf, rand_base)
            assert ring_score(seq, **kwargs) == pytest.approx(0.0, abs=delta), msg

    def test_infiniteness(self, rand_base, rand_zero_pads):
        seq = np.random.uniform(size=5)
        for ring_score in INF_RING_SCORES - {entropy_ring_score}:
            msg = _generate_message(seq, ring_score, np.inf, rand_base)
            kwargs = {}
            if "base" in inspect.signature(ring_score).parameters:
                kwargs["base"] = rand_base

            score1 = ring_score(seq, nb_pers=np.inf, **kwargs)
            score2 = ring_score(
                list(seq) + [0] * rand_zero_pads, nb_pers=np.inf, **kwargs
            )

            assert score1 == pytest.approx(score2), msg


@pytest.fixture
def pseq_list():
    return [np.random.uniform(size=N) for _ in range(2**2) for N in range(2, 2**2)]


@pytest.fixture
def p0_list(pseq_list):
    return [sorted(pseq)[-1] for pseq in pseq_list]


@pytest.fixture
def p1_list(pseq_list):
    return [sorted(pseq)[-2] for pseq in pseq_list]


@pytest.fixture
def gap_scores(pseq_list):
    return [gap_ring_score(pseq) for pseq in pseq_list]


@pytest.fixture
def amp_scores(pseq_list):
    return [amplitude_ring_score(pseq, nb_pers=2) for pseq in pseq_list]


@pytest.fixture
def ent_scores(pseq_list):
    return [entropy_ring_score(pseq, nb_pers=2) for pseq in pseq_list]


class TestTwoPersistenceIdentities:
    def test_gap_identities(self, p0_list, p1_list, gap_scores):
        test_scores1 = [1 - p1 / p0 for (p0, p1) in zip(p0_list, p1_list)]

        assert assert_allclose(gap_scores, test_scores1) is None

    def test_amp_identities(self, p0_list, p1_list, gap_scores, amp_scores):
        test_scores1 = [(p0 - p1) / (p0 + p1) for (p0, p1) in zip(p0_list, p1_list)]
        test_scores2 = [gap / (2 - gap) for gap in gap_scores]

        assert assert_allclose(amp_scores, test_scores1) is None
        assert assert_allclose(amp_scores, test_scores2) is None

    def test_ent_identities(self, p0_list, p1_list, amp_scores, ent_scores):
        def p1_over_p0_identity(p):
            numer = (p + 1) * np.log(p + 1) - p * np.log(p)
            denom = (p + 1) * np.log(2)
            return 1 - numer / denom

        def amp_identity(x):
            numer = 2 * x * np.arctanh(x) + np.log(1 - x**2)
            denom = np.log(4)
            return numer / denom

        test_scores1 = [
            p1_over_p0_identity(p0 / p1) for (p0, p1) in zip(p0_list, p1_list)
        ]
        test_scores2 = [amp_identity(amp) for amp in amp_scores]

        assert assert_allclose(ent_scores, test_scores1) is None
        assert assert_allclose(ent_scores, test_scores2) is None
