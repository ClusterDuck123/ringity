"""Plot angle timeseries for focal genes, coloured by circadian phase.

When genes in the angles file have biological mRNA peak times defined in BIO_ZT,
two additional panels are added: a ZT assignment strip and a circular correlation
scatter with permutation p-value.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from pathlib import Path

TWO_PI = 2 * np.pi

PHASE_COLORS = {
    "morning": "#E07B22",
    "day":     "#B8A800",
    "evening": "#7B4EA0",
    "night":   "#1A3A5C",
}

PHASE_SPANS = [          # (phase, zt_start, zt_end) — for background shading
    ("morning",  0,  6),
    ("day",      6, 12),
    ("evening", 12, 18),
    ("night",   18, 24),
]

MIN_LABEL_GAP = 0.12

# mRNA peak ZT times from primary literature (see circadian_phase_citations.md).
# ZTL and LKP2 omitted from BIO_ZT: constitutive mRNA, no transcript peak to assign.
BIO_ZT = {
    "CCA1": 1.0,    # ZT0–2,   Wang & Tobin 1998
    "LHY":  1.0,    # ZT0–2,   Schaffer 1998
    "LNK1": 1.75,   # ZT1.5–2, Rugnone 2013
    "LNK2": 1.75,   # ZT1.5–2, Rugnone 2013
    "PRR7": 6.0,    # ZT4–8,   Nakamichi 2010
    "FKF1": 8.0,    # ~ZT8,    Nelson 2000
    "PRR5": 9.0,    # ZT8–10,  Nakamichi 2010
    "GI":   9.0,    # ZT8–10,  Fowler 1999
    "CHE":  9.0,    # ~ZT9,    Pruneda-Paz 2009
    "PRR3": 11.0,   # ZT10–12, Nakamichi 2010
    "ELF4": 12.0,   # ~ZT12,   Doyle 2002
    "TOC1": 13.0,   # ZT12–14, Strayer 2000
    "ELF3": 15.0,   # ZT14–16, Liu 2001
}

# Protein-stability peak ZT for genes with constitutive mRNA.
# Shown on the ZT strip for completeness but excluded from the correlation.
PROTEIN_ZT = {
    "ZTL":  18.0,   # protein peaks late afternoon/early night, Kim 2007
    "LKP2": 20.0,   # structurally similar to ZTL; protein oscillation uncharacterised
}

# Protein-phase peaks for ALL genes (--protein mode).
# Where protein timing is not separately characterised, mRNA timing is used.
PROTEIN_ZT_ALL = {
    "CCA1": 1.0,    # protein at dawn, same as mRNA; Wang & Tobin 1998
    "LHY":  1.0,    # protein at dawn, same as mRNA; Schaffer 1998
    "LNK1": 2.0,    # protein timing not well separated from mRNA (~ZT2)
    "LNK2": 2.0,    # same
    "PRR7": 7.0,    # protein slightly trails mRNA (ZT4–8 → ~ZT7)
    "PRR5": 10.0,   # protein ZT10–12; Nakamichi 2010
    "PRR3": 12.0,   # protein slightly later than mRNA (ZT11 → ZT12)
    "GI":   14.0,   # GI protein accumulates in afternoon/dusk; David 2006
    "FKF1": 14.0,   # FKF1 protein stabilised by light, active at dusk; Imaizumi 2003
    "CHE":  11.0,   # protein ZT9–13, midpoint ~ZT11; Kamioka 2016
    "TOC1": 14.0,   # protein peak ~ZT14 before ZTL-mediated degradation; Más 2003
    "ELF4": 13.0,   # protein peak slightly after mRNA (~ZT13)
    "ELF3": 14.0,   # protein at dusk ~ZT12–16; Liu 2001
    "ZTL":  18.0,   # protein stability peak late afternoon/night; Kim 2007
    "LKP2": 20.0,   # structural analogue of ZTL
}


# ---------------------------------------------------------------------------
# Circular statistics
# ---------------------------------------------------------------------------

def _circular_mean(angles):
    return np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))


def _jammalamadaka_r(alpha, beta):
    a_bar = _circular_mean(alpha)
    b_bar = _circular_mean(beta)
    num   = np.sum(np.sin(alpha - a_bar) * np.sin(beta - b_bar))
    denom = np.sqrt(
        np.sum(np.sin(alpha - a_bar) ** 2) *
        np.sum(np.sin(beta  - b_bar) ** 2)
    )
    return num / denom


def _permutation_pvalue(bio_angles, net_angles, r_obs, n_perm=10_000, seed=0):
    rng    = np.random.default_rng(seed)
    r_perm = np.array([
        _jammalamadaka_r(bio_angles, rng.permutation(net_angles))
        for _ in range(n_perm)
    ])
    return np.mean(np.abs(r_perm) >= np.abs(r_obs))


# ---------------------------------------------------------------------------
# Wrap interpolation & label staggering
# ---------------------------------------------------------------------------

def interpolate_wraps(t, arr):
    """Split a wrapped angle series at each 0/2π crossing, extending each
    segment to the exact interpolated boundary crossing time."""
    arr   = arr.astype(float)
    jumps = np.where(np.abs(np.diff(arr)) > np.pi)[0]
    if len(jumps) == 0:
        return t.copy(), arr.copy()
    t_out, v_out = [], []
    prev = 0
    for i in jumps:
        v0, v1 = arr[i], arr[i + 1]
        t0, t1 = t[i], t[i + 1]
        if v1 < v0:
            v1_unwrapped  = v1 + TWO_PI
            boundary_pre  = TWO_PI
            boundary_post = 0.0
        else:
            v1_unwrapped  = v1 - TWO_PI
            boundary_pre  = 0.0
            boundary_post = TWO_PI
        t_cross = t0 + (boundary_pre - v0) / (v1_unwrapped - v0) * (t1 - t0)
        t_out.extend(t[prev : i + 1].tolist() + [t_cross, np.nan])
        v_out.extend(arr[prev : i + 1].tolist() + [boundary_pre, np.nan])
        t_out.append(t_cross)
        v_out.append(boundary_post)
        prev = i + 1
    t_out.extend(t[prev:].tolist())
    v_out.extend(arr[prev:].tolist())
    return np.array(t_out), np.array(v_out)


def stagger_labels(positions):
    """Nudge a sorted list of y-positions apart so no two are closer than MIN_LABEL_GAP."""
    pos = sorted(positions)
    for _ in range(200):
        changed = False
        for i in range(1, len(pos)):
            gap = pos[i] - pos[i - 1]
            if gap < MIN_LABEL_GAP:
                shift = (MIN_LABEL_GAP - gap) / 2
                pos[i - 1] -= shift
                pos[i]     += shift
                changed = True
        if not changed:
            break
    return pos


# ---------------------------------------------------------------------------
# Extra panels
# ---------------------------------------------------------------------------

def _draw_zt_strip(ax, symbols, phase_of, zt_dict, protein_mode=False):
    """Horizontal strip: dot per gene at its peak ZT, colored by class.

    In mRNA mode (protein_mode=False): genes absent from zt_dict (ZTL, LKP2)
    are shown as open circles from PROTEIN_ZT with a footnote explaining
    they are excluded from the correlation.
    In protein mode (protein_mode=True): all genes are filled circles.
    """
    for phase, zt0, zt1 in PHASE_SPANS:
        ax.axvspan(zt0, zt1, color=PHASE_COLORS[phase], alpha=0.13, lw=0)

    groups = defaultdict(list)
    for sym in symbols:
        groups[zt_dict[sym]].append(sym)

    DOT_Y = 0.65
    for zt, group in sorted(groups.items()):
        n      = len(group)
        x_offs = np.linspace(-0.3 * (n - 1), 0.3 * (n - 1), n)
        y_dots = np.linspace(DOT_Y + 0.12 * (n - 1),
                             DOT_Y - 0.12 * (n - 1), n)
        for sym, dx, dy in zip(group, x_offs, y_dots):
            color = PHASE_COLORS[phase_of[sym]]
            ax.plot(zt + dx, dy, 'o', color=color, ms=6, zorder=3, clip_on=False)
            ax.text(zt + dx, dy - 0.07, sym,
                    color=color, fontsize=6.5,
                    ha='center', va='top', rotation=90, clip_on=False)

    if not protein_mode:
        # Show genes with constitutive mRNA as open circles at their protein peak
        protein_in_data = [s for s in PROTEIN_ZT if s in phase_of]
        for sym in protein_in_data:
            zt    = PROTEIN_ZT[sym]
            color = PHASE_COLORS[phase_of[sym]]
            ax.plot(zt, DOT_Y, 'o', color=color, ms=6, zorder=3,
                    mfc='white', mew=1.5, clip_on=False)
            ax.text(zt, DOT_Y - 0.07, sym + "†",
                    color=color, fontsize=6.5,
                    ha='center', va='top', rotation=90, clip_on=False)
        if protein_in_data:
            ax.text(0.01, 0.01,
                    "† protein-phase peak shown (mRNA constitutive);\n"
                    "  excluded from correlation and p-value",
                    transform=ax.transAxes, fontsize=6,
                    va='bottom', color='#555555', style='italic')

    title = ("Protein peak times (literature)"
             if protein_mode else "Biological peak times (literature)")
    ax.set_xlim(-0.5, 24.5)
    ax.set_ylim(-1.4, 1.0)
    ax.set_xlabel("Time of day (ZT hours)", fontsize=9)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_yticks([])
    for sp in ('left', 'right', 'top'):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=9)


def _draw_corr_scatter(ax, symbols, net_angles, phase_of, r, p, zt_dict,
                       xlabel="Biological mRNA peak (ZT h)"):
    """Scatter: peak ZT (x) vs network phase in ZT-equivalent h (y)."""
    bio_arr = np.array([zt_dict[s]         for s in symbols])
    net_arr = np.array(net_angles) / TWO_PI * 24   # rad → ZT-equivalent hours

    for sym, bzt, nzt in zip(symbols, bio_arr, net_arr):
        color = PHASE_COLORS[phase_of[sym]]
        ax.scatter(bzt, nzt, color=color, s=50, zorder=3)
        ax.text(bzt + 0.4, nzt, sym, color=color, fontsize=7, va='center')

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Network phase  (ZT-equiv. h)", fontsize=9)
    ax.set_xlim(-1, 27)
    ax.set_ylim(-1, 27)
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_yticks([0, 6, 12, 18, 24])

    sign = "−" if r < 0 else "+"
    ax.text(0.05, 0.97,
            f"r = {sign}{abs(r):.3f}\np = {p:.3f}  (two-tailed)",
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#aaaaaa', alpha=0.9))
    ax.set_title("Circular correlation\n(Jammalamadaka–SenGupta)", fontsize=9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(angles_file, output_file, protein_mode=False):
    with open(angles_file) as f:
        data = json.load(f)

    t        = np.array(data["t"])
    natfreq  = data["params"].get("natfreq", 1.0)
    phase_of = data["phase_of"]
    angles   = data["angles"]
    t_final  = t[-1]

    zt_dict  = PROTEIN_ZT_ALL if protein_mode else BIO_ZT
    bio_syms = [s for s in angles if s in zt_dict]
    has_bio  = len(bio_syms) >= 3

    # Determine canonical orientation: reflect all angles if network settles
    # anti-correlated with biological phase (Kuramoto rotational symmetry).
    if has_bio:
        net_finals_bio = [(angles[s][-1] - natfreq * t_final) % TWO_PI
                          for s in bio_syms]
        bio_angles_chk = np.array([zt_dict[s] / 24 * TWO_PI for s in bio_syms])
        r_chk = _jammalamadaka_r(bio_angles_chk, np.array(net_finals_bio))
        reflect = r_chk < 0
    else:
        reflect = False

    # ---- figure layout ----
    if has_bio:
        fig = plt.figure(figsize=(14, 12))
        gs  = gridspec.GridSpec(2, 2, height_ratios=[3, 2.5],
                                hspace=0.38, wspace=0.32, figure=fig)
        ax_time  = fig.add_subplot(gs[0, :])
        ax_strip = fig.add_subplot(gs[1, 0])
        ax_corr  = fig.add_subplot(gs[1, 1])
    else:
        fig, ax_time = plt.subplots(figsize=(10, 5))

    # ---- timeseries panel ----
    series      = {}
    finals      = {}
    seen_phases = set()

    for symbol, angle_series in angles.items():
        detrended = (np.array(angle_series) - natfreq * t) % TWO_PI
        if reflect:
            detrended = (TWO_PI - detrended) % TWO_PI
        finals[symbol] = detrended[-1]
        series[symbol] = interpolate_wraps(t, detrended)

        phase = phase_of[symbol]
        color = PHASE_COLORS.get(phase, "grey")
        label = phase if phase not in seen_phases else None
        seen_phases.add(phase)

        t_plot, v_plot = series[symbol]
        ax_time.plot(t_plot, v_plot, color=color, label=label,
                     linewidth=1.5, alpha=0.85)

    symbols_sorted = sorted(finals, key=finals.get)
    nudged_y       = stagger_labels([finals[s] for s in symbols_sorted])
    for symbol, y in zip(symbols_sorted, nudged_y):
        phase = phase_of[symbol]
        color = PHASE_COLORS.get(phase, "grey")
        ax_time.annotate(
            symbol,
            xy=(t[-1], finals[symbol]),
            xytext=(t[-1] + (t[-1] - t[0]) * 0.01, y),
            xycoords="data", textcoords="data",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.5),
            color=color, fontsize=7.5, va="center", annotation_clip=False,
        )

    ax_time.set_xlim(t[0], t[-1])
    ax_time.set_xlabel("Time")
    ax_time.set_ylabel("(θ − ωt)  mod  2π  [rad]")
    ax_time.set_ylim(0, TWO_PI)
    ax_time.set_yticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, TWO_PI])
    ax_time.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax_time.legend(title="Phase", loc="upper left", framealpha=0.7)

    # ---- panel labels ----
    if has_bio:
        for ax, label in [(ax_time, "A"), (ax_strip, "B"), (ax_corr, "C")]:
            ax.text(-0.04, 1.04, label, transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="bottom", ha="right")

    # ---- extra panels ----
    if has_bio:
        _draw_zt_strip(ax_strip, bio_syms, phase_of, zt_dict,
                       protein_mode=protein_mode)

        net_finals     = [(angles[s][-1] - natfreq * t_final) % TWO_PI
                          for s in bio_syms]
        bio_angles     = np.array([zt_dict[s] / 24 * TWO_PI for s in bio_syms])
        net_angles_arr = np.array(net_finals)

        r_obs = _jammalamadaka_r(bio_angles, net_angles_arr)
        p_val = _permutation_pvalue(bio_angles, net_angles_arr, r_obs)

        # The Kuramoto model has rotational symmetry (uniform ω, random init),
        # so both orientations of the circle are equivalent solutions.  Choose
        # the canonical orientation that gives positive correlation with the
        # biological phase; reflect angles as (2π − θ) if r < 0.
        if r_obs < 0:
            net_finals     = [(TWO_PI - v) % TWO_PI for v in net_finals]
            net_angles_arr = TWO_PI - net_angles_arr
            r_obs          = abs(r_obs)

        xlabel = ("Protein peak (ZT h)"
                  if protein_mode else "Biological mRNA peak (ZT h)")
        _draw_corr_scatter(ax_corr, bio_syms, net_finals, phase_of,
                           r_obs, p_val, zt_dict, xlabel=xlabel)

    fig.tight_layout()

    if output_file is None:
        output_file = Path(angles_file).with_suffix(".png")
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("angles_file", nargs="?",
                        default=str(Path(__file__).parent / "arabidopsis_angles.json"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--protein", action="store_true",
                        help="Use protein-phase peaks instead of mRNA peaks")
    args = parser.parse_args()
    main(args.angles_file, args.output, protein_mode=args.protein)
