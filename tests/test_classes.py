import random
import numpy as np

from ringity.tda.pdiagram.generators import random_pdgm_point, random_pdgm
from ringity.tda.pdiagram import pdiagram

import pytest


@pytest.fixture
def number_and_dgm_pt():
    rng = np.random.default_rng()
    number = rng.uniform()
    dgm_pt = random_pdgm_point()
    return number, dgm_pt


@pytest.fixture
def signal_and_noise_pt():
    rng = np.random.default_rng()
    number = rng.uniform()
    signal_pt = pdiagram.PDiagramPoint((0, number))
    noise_pt = pdiagram.PDiagramPoint((0, 0))
    return signal_pt, noise_pt


class TestDgmPt:
    def test_arithmetic_operations(self, number_and_dgm_pt):
        number, dgm_pt = number_and_dgm_pt
        np.testing.assert_allclose(np.array(dgm_pt + number), np.array(dgm_pt) + number)
        np.testing.assert_allclose(np.array(dgm_pt / number), np.array(dgm_pt) / number)


class TestDgm:
    def test_score1(self, signal_and_noise_pt):
        signal_pt, noise_pt = signal_and_noise_pt
        dgm = pdiagram.PDiagram(signal_pt for _ in range(100))
        assert dgm.ring_score() == pytest.approx(0, abs=1e-8)

    def test_score2(self, signal_and_noise_pt):
        signal_pt, noise_pt = signal_and_noise_pt
        dgm = pdiagram.PDiagram(noise_pt for _ in range(100))
        dgm.append(signal_pt)
        assert dgm.ring_score() == pytest.approx(1, abs=1e-8)

    def test_birth_and_death_extractions(self):
        dgm = random_pdgm(100)
        assert tuple(dgm.births) == tuple(pt.birth for pt in dgm)
        assert tuple(dgm.deaths) == tuple(pt.death for pt in dgm)
