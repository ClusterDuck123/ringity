import pytest
import numpy as np
import ringity.tda.pdiagram as pdgm

from pathlib import Path
from ringity.ringscore.metric2ringscore import ring_score_from_sequence

DIRNAME_TEST_DATA = Path(__file__).parent / "test_data"
FNAME_PDGM = DIRNAME_TEST_DATA / "lipid_coexp_dgm.txt"

EXPECTED_LIPID_RING_SCORE = 0.7669806588679011


@pytest.fixture
def signal_and_noise():
    rng = np.random.default_rng()
    signal = rng.uniform()
    noise = [0] * rng.integers(0, 10)
    return signal, noise


class TestSyntheticExamples:
    def test_score_of_zero1(self, signal_and_noise):
        signal, noise = signal_and_noise
        multiple_signals = [signal] * 50
        sequence = noise + multiple_signals + noise + multiple_signals + noise
        assert ring_score_from_sequence(sequence) == pytest.approx(0, abs=1e-8)

    def test_score_of_zero2(self):
        sequence = ()
        assert ring_score_from_sequence(sequence) == pytest.approx(0, abs=1e-8)

    def test_score_of_one(self, signal_and_noise):
        signal, noise = signal_and_noise
        sequence = noise + [signal] + noise
        assert ring_score_from_sequence(sequence) == pytest.approx(1, abs=1e-8)

    def test_score_of_half(self, signal_and_noise):
        signal, noise = signal_and_noise
        sequence = noise + [signal] + noise + [signal] + noise
        assert ring_score_from_sequence(sequence) == pytest.approx(0.5, abs=1e-8)

    def test_score_of_two_high_noises(self, signal_and_noise):
        signal, noise = signal_and_noise
        noise_ratio2, noise_ratio1 = sorted(np.random.uniform(size=2))

        noise1 = signal * noise_ratio1
        noise2 = signal * noise_ratio2

        sequence = noise + [signal] + noise + [noise1] + noise + [noise2] + noise
        assert ring_score_from_sequence(sequence) == pytest.approx(
            1 - noise_ratio1 / 2 - noise_ratio2 / 4, abs=1e-8
        )


class TestKnownNetworks:
    def test_lipid_network(self):
        dgm = pdgm.readwrite.read_pdiagram(FNAME_PDGM)
        assert dgm.ring_score() == pytest.approx(EXPECTED_LIPID_RING_SCORE)
