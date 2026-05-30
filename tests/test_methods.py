import os
import ringity.tda.pdiagram as pdgm
import pytest

from pathlib import Path
from ringity.ringscore.metric2ringscore import (
    gap_ring_score,
    geometric_ring_score,
    linear_ring_score,
    amplitude_ring_score,
    entropy_ring_score,
)
from ringity.tda.pdiagram.generators import random_pdgm
from ringity.tda.pdiagram.readwrite import write_pdiagram, read_pdiagram

DIRNAME_TMP = Path(__file__).parent / "test_data" / "tmp"
FNAME_PDGM = DIRNAME_TMP / "random_dgm.txt"


class TestReadAndWrite:
    def test_save_and_load(self):
        pdgm1 = random_pdgm(2**5)
        pdgm.readwrite.write_pdiagram(pdgm1, FNAME_PDGM)

        pdgm2 = pdgm.readwrite.read_pdiagram(FNAME_PDGM)
        os.remove(FNAME_PDGM)

        assert pdgm1 == pdgm2


class TestRingScore:
    def test_ring_score_flavours(self):
        pdgm = random_pdgm(2**5)
        pseq = pdgm.psequence(normalisation="signal")

        assert gap_ring_score(pseq) == pytest.approx(
            pdgm.ring_score(flavour="gap"),
        )
        assert linear_ring_score(pseq, nb_pers=4) == pytest.approx(
            pdgm.ring_score(flavour="linear", nb_pers=4),
        )
        assert geometric_ring_score(pseq, nb_pers=4, exponent=3) == pytest.approx(
            pdgm.ring_score(flavour="geometric", nb_pers=4, exponent=3),
        )
        assert amplitude_ring_score(pseq, nb_pers=4) == pytest.approx(
            pdgm.ring_score(flavour="amplitude", nb_pers=4),
        )
        assert entropy_ring_score(pseq, nb_pers=4) == pytest.approx(
            pdgm.ring_score(flavour="entropy", nb_pers=4),
        )
        assert geometric_ring_score(pseq, nb_pers=4) != pytest.approx(
            pdgm.ring_score(flavour="geometric", nb_pers=4, exponent=3),
        )
