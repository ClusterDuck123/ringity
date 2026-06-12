"""Circular-circular correlation between network-derived phase and biological mRNA peak.

Uses the Jammalamadaka-SenGupta coefficient with a permutation test.
ZTL and LKP2 are excluded (constitutive mRNA, no peak to assign).
"""

import json
import numpy as np
from pathlib import Path

TWO_PI = 2 * np.pi

# mRNA peak ZT times from primary literature (see circadian_phase_citations.md)
BIO_ZT = {
    "CCA1":  1.0,   # ZT0-2, Wang & Tobin 1998
    "LHY":   1.0,   # ZT0-2, Schaffer 1998
    "LNK1":  1.75,  # ZT1.5-2, Rugnone 2013
    "LNK2":  1.75,  # ZT1.5-2, Rugnone 2013
    "PRR7":  6.0,   # ZT4-8, Nakamichi 2010
    "PRR5":  9.0,   # ZT8-10, Nakamichi 2010
    "PRR3":  11.0,  # ZT10-12, Nakamichi 2010
    "GI":    9.0,   # ZT8-10, Fowler 1999
    "FKF1":  8.0,   # ~ZT8, Nelson 2000
    "CHE":   9.0,   # ~ZT9 (9h out of phase with CCA1), Pruneda-Paz 2009
    "TOC1":  13.0,  # ZT12-14, Strayer 2000
    "ELF4":  12.0,  # ~ZT12, Doyle 2002
    "ELF3":  15.0,  # ZT14-16, Liu 2001
}

BIO_ANGLES = {sym: zt / 24 * TWO_PI for sym, zt in BIO_ZT.items()}


def circular_mean(angles):
    return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))


def jammalamadaka_correlation(alpha, beta):
    """Jammalamadaka-SenGupta circular correlation coefficient."""
    a_bar = circular_mean(alpha)
    b_bar = circular_mean(beta)
    num   = np.sum(np.sin(alpha - a_bar) * np.sin(beta - b_bar))
    denom = np.sqrt(
        np.sum(np.sin(alpha - a_bar) ** 2) *
        np.sum(np.sin(beta  - b_bar) ** 2)
    )
    return num / denom


def main(angles_file="arabidopsis_angles_clean_night.json", n_perm=10_000, seed=0):
    with open(Path(__file__).parent / angles_file) as f:
        data = json.load(f)

    t      = np.array(data["t"])
    angles = data["angles"]

    symbols = [s for s in angles if s in BIO_ANGLES]
    missing = [s for s in angles if s not in BIO_ANGLES]
    if missing:
        print(f"Skipped (no mRNA peak assigned): {', '.join(missing)}")

    natfreq = data["params"].get("natfreq", 1.0)
    t_final = t[-1]

    bio  = np.array([BIO_ANGLES[s] for s in symbols])
    # detrend exactly as the plot does: subtract ω·t then wrap to [0, 2π)
    net  = np.array([(angles[s][-1] - natfreq * t_final) % TWO_PI for s in symbols])

    r_obs = jammalamadaka_correlation(bio, net)
    print(f"\nn = {len(symbols)} genes")
    print(f"Observed JS circular correlation: r = {r_obs:.4f}")

    rng    = np.random.default_rng(seed)
    r_perm = np.array([
        jammalamadaka_correlation(bio, rng.permutation(net))
        for _ in range(n_perm)
    ])

    # Two-tailed on |r|: the circle has no preferred orientation, so a strong
    # negative r (reversed ordering) is as informative as a strong positive r.
    p = np.mean(np.abs(r_perm) >= np.abs(r_obs))
    print(f"Permutation p-value (two-tailed, |r| >= |r_obs|): p = {p:.4f}  ({n_perm} permutations)")
    print(f"(Note: r < 0 means the network settled with evening at low phase, morning at")
    print(f" high phase — both orientations are equivalent on a circle.)")

    print(f"\nPer-gene breakdown (sorted by network phase, low → high):")
    print(f"  {'gene':6s}  {'bio ZT':>6s}  {'bio angle':>9s}  {'net angle':>9s}")
    for i in np.argsort(net):
        s = symbols[i]
        print(f"  {s:6s}  {BIO_ZT[s]:6.2f}h  {bio[i]:.4f} rad  {net[i]:.4f} rad")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--angles_file", default="arabidopsis_angles_clean_night.json")
    parser.add_argument("--n_perm", type=int, default=10_000)
    args = parser.parse_args()
    main(args.angles_file, args.n_perm)
