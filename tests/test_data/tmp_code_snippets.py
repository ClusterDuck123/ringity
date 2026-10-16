"""Use this script for quick functionality tests or bug fixes."""

import ringity as rty
from ringity.core.metric2ringscore import gap_ring_score

pdgm = rty.random_pdgm(2**5)
pseq = pdgm.psequence(normalisation="signal")

print(gap_ring_score(pseq))
print(pdgm.ring_score(flavour="gap"))
