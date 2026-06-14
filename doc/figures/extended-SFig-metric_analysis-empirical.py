# %% [markdown]
# # Metric-sensitivity of the empirical ring scores
#
# Reviewer #1 asks how sensitive the empirical case studies are to *reasonable alternative
# preprocessing choices*. A central such choice for an unweighted network is the
# **graph → metric construction**: how nodes are equipped with pairwise distances before
# persistent homology. This script reproduces, for the five empirical networks in
# `doc/data/empirical_networks/`, the metric comparison the author ran for synthetic
# Watts–Strogatz graphs in
# `doc/figures/extended-SFigX-exploring_different_metric_structures.ipynb`.
#
# For every network we compute the ring score under four metrics —
# `spl` (shortest-path), `resistance` distance, a `betweenness`-induced metric, and
# `current_flow` (the random-walk net-flow used throughout the paper; ring score is
# scale-invariant, so `current_flow` and `net_flow` give the same score). The figure shows
# how each network's ring score moves as the metric changes: a flat, high line means the
# ring signal is **robust** to the construction; a steep line means it is **metric-sensitive**.
# `current_flow` is the construction validated on known ring topologies (Extended Data Fig. 1).
#
# **Format note.** Percent-format (`# %%`) script — runs cell-by-cell in the VS Code Python
# Interactive Window and as `python extended-SFig-metric_analysis-empirical.py`.
#
# **Cost.** Four of the networks are tiny (<2 s for all metrics); the gene network
# (7,612 nodes) is expensive (~45–90 min, Ripser-dominated). Scores are cached incrementally
# to CSV, so the gene cost is paid once and re-runs/plots are instant. Set the environment
# variable `SKIP_NETWORKS=gene` for a fast dry run on the four small networks.

# %%
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import ringity as rty

rty.set_theme()

# CEMM accents (mirrors src/ringity/utils/plotting/styling.py)
CEMM_DARK = (0 / 255, 43 / 255, 50 / 255)
CEMM_2 = (0 / 255, 140 / 255, 160 / 255)
CEMM_4 = (212 / 255, 236 / 255, 242 / 255)

# Networks (filename stem → Fig. 1 display label), in a stable plotting order.
NETWORKS = [
    ("lipid", "Lipid co-regulation"),
    ("gene", "Gene co-expression"),
    ("immune", "Immune signaling"),
    ("fibro", "Fibroblast proximity"),
    ("soil", "Soil moisture"),
]
NET_COLORS = {
    "lipid": "#1f77b4",
    "gene": "#ff7f0e",
    "immune": "#2ca02c",
    "fibro": "#9467bd",
    "soil": "#8c564b",
}

# Metrics in increasing faithfulness, ending on the paper's chosen construction.
METRIC_ORDER = ["spl", "resistance", "betweenness", "current_flow"]
METRIC_LABELS = {
    "spl": "shortest\npath",
    "resistance": "resistance\ndistance",
    "betweenness": "betweenness",
    "current_flow": "current\nflow",
}

SKIP_NETWORKS = set(filter(None, os.environ.get("SKIP_NETWORKS", "").split(",")))


# %%
# --- Path handling (anchor on this file's location, robust to interactive CWD) -------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()


def _find_data_dir():
    for cand in (
        BASE_DIR / ".." / "data",
        Path.cwd() / "doc" / "data",
        Path.cwd() / ".." / "data",
        Path.cwd() / "data",
    ):
        cand = cand.resolve()
        if (cand / "empirical_networks").exists():
            return cand
    return (BASE_DIR / ".." / "data").resolve()


DATA_DIR = _find_data_dir()
EMP_DIR = DATA_DIR / "empirical_networks"
RESULTS_DIR = DATA_DIR / "results"
PLOTS_DIR = DATA_DIR / "plots"
CACHE_FILE = RESULTS_DIR / "ringscores_empirical_metrics.csv"
FIG_STEM = PLOTS_DIR / "SFig-metric_analysis-empirical-summary"

if not EMP_DIR.exists():
    raise FileNotFoundError(f"Empirical networks not found at {EMP_DIR}.")

print(f"EMP_DIR: {EMP_DIR}  ({len(list(EMP_DIR.glob('*.gml')))} networks)")


# %% [markdown]
# ## Ring score per network × metric (incremental cache)
#
# `compute_metric_scores` loads the CSV, computes only the missing `(network, metric)` pairs,
# and saves after *each* computation — so the long gene run is resumable and never lost.


# %%
def compute_metric_scores(networks, metrics, emp_dir, cache_file, skip=()):
    cols = ["network", "metric", "n_nodes", "n_edges", "ring_score"]
    if cache_file.exists():
        df = pd.read_csv(cache_file)
    else:
        df = pd.DataFrame(columns=cols)
    done = {(r.network, r.metric) for r in df.itertuples()}

    for stem, _label in networks:
        if stem in skip:
            continue
        G = None  # loaded lazily, only if a metric is missing
        for metric in metrics:
            if (stem, metric) in done:
                continue
            if G is None:
                G = nx.read_gml(emp_dir / f"{stem}.gml")
            t = time.time()
            score = rty.pdiagram_from_network(G, metric=metric).ring_score()
            df.loc[len(df)] = [
                stem,
                metric,
                G.number_of_nodes(),
                G.number_of_edges(),
                score,
            ]
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_file, index=False)
            print(
                f"  {stem:8} {metric:13} S={score:.3f}  ({time.time() - t:.1f}s)",
                flush=True,
            )
    return df


# %%
df = compute_metric_scores(
    NETWORKS, METRIC_ORDER, EMP_DIR, CACHE_FILE, skip=SKIP_NETWORKS
)

# Wide table for a quick read: rows = networks, cols = metrics.
table = (
    df.pivot(index="network", columns="metric", values="ring_score")
    .reindex([s for s, _ in NETWORKS])
    .reindex(columns=METRIC_ORDER)
)
print(table.round(3).to_string())


# %% [markdown]
# ## Figure — slopegraph of ring score across metrics
#
# One line per empirical network across the four metrics; the `current_flow` column (the
# paper's validated construction) is highlighted. Lines that stay high are robust to the
# metric choice; lines that drop are metric-sensitive.


# %%
def _spread_labels(values, min_gap=0.05, top=1.06):
    """Greedy vertical declutter for the right-hand slopegraph labels."""
    order = np.argsort(values)
    pos = np.array(values, dtype=float)
    for i in range(1, len(order)):
        lo, hi = order[i - 1], order[i]
        if pos[hi] - pos[lo] < min_gap:
            pos[hi] = pos[lo] + min_gap
    # if the stack ran past the top, shift the whole thing down
    overflow = pos.max() - top
    if overflow > 0:
        pos -= overflow
    return pos


def make_figure(df):
    present = [(s, lab) for s, lab in NETWORKS if s in set(df["network"])]
    scores = {
        s: df[df["network"] == s].set_index("metric")["ring_score"].to_dict()
        for s, _ in present
    }

    x = np.arange(len(METRIC_ORDER))
    last = x[-1]
    fig, ax = plt.subplots(figsize=(11, 8))

    # highlight the chosen-metric column (label sits low, where no score falls)
    ax.axvspan(last - 0.3, last + 0.3, color=CEMM_4, alpha=0.7, zorder=0)
    ax.text(
        last,
        0.28,
        "metric used\nin the paper",
        ha="center",
        va="center",
        fontsize=12,
        color=CEMM_DARK,
        fontweight="bold",
        zorder=4,
    )

    # one line per network
    end_vals = []
    for stem, label in present:
        y = [scores[stem].get(m, np.nan) for m in METRIC_ORDER]
        ax.plot(
            x,
            y,
            "-o",
            color=NET_COLORS[stem],
            lw=2.5,
            ms=9,
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=3,
        )
        end_vals.append((stem, label, y[-1]))

    # decluttered right-hand labels: "Name (score)"
    lab_y = _spread_labels([v for *_, v in end_vals])
    for (stem, label, yv), ly in zip(end_vals, lab_y):
        if abs(ly - yv) > 1e-3:  # faint leader when the label was nudged
            ax.plot(
                [last, last + 0.18],
                [yv, ly],
                color=NET_COLORS[stem],
                lw=0.8,
                alpha=0.6,
                zorder=2,
            )
        ax.text(
            last + 0.22,
            ly,
            f"{label}  ({yv:.2f})",
            va="center",
            ha="left",
            fontsize=12,
            color=NET_COLORS[stem],
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRIC_ORDER])
    ax.set_xlim(-0.3, last + 1.7)
    ax.set_ylim(-0.02, 1.09)
    ax.set_ylabel("ring score $S$")
    ax.set_xlabel("graph → metric construction")
    ax.set_title(
        "Sensitivity of empirical ring scores to the metric construction",
        fontsize="large",
        fontweight="bold",
        loc="left",
    )
    return fig


fig = make_figure(df)
fig

# %%
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_STEM.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
fig.savefig(FIG_STEM.with_suffix(".png"), bbox_inches="tight", dpi=200)
print(
    f"Saved {FIG_STEM.with_suffix('.pdf').name} and {FIG_STEM.with_suffix('.png').name}"
)

# %% [markdown]
# ### Reading the figure for the rebuttal
#
# - The ring score is **not** a fixed property of a network — it depends on the graph→metric
#   construction. `shortest-path` and `resistance` distances collapse the signal (they fail to
#   resolve global cycles, as also shown on synthetic graphs with *known* ring topology in
#   Extended Data Fig. 1), whereas the random-walk `current_flow` metric recovers it.
# - Under the validated `current_flow` construction, the strong empirical ring scores
#   (e.g. lipid ≈ 0.97, soil ≈ 1.0, gene ≈ 0.82) are recovered; networks differ in how
#   sensitive they are to the choice, which this figure makes explicit.
# - This addresses the metric-construction axis of the reviewer's preprocessing-sensitivity
#   concern; weighting/correlation-to-edge/threshold sensitivity is a complementary axis
#   (partly already reported in the manuscript Methods).
