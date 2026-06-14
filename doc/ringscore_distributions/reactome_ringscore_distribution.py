# %% [markdown]
# # Ring-score distribution across Reactome pathways
#
# This file addresses the reviewer request to characterise *how the ring score
# `S` is distributed across the Reactome pathways*, rather than only reporting
# the handful of pathways above the `S > 0.6` threshold.
#
# Concretely, the four-panel figure answers the points raised in review:
#
# - **(a)** distribution of `S` across all networks with at least `MIN_NODES`
#   nodes (not just the ringy ones) — a dominant ring is the exception.
# - **(b)** the networks with `S > 0.6`, named and ranked.
# - **(c)** `S` vs. network size — the ring signal is not a size artifact.
# - **(d)** clustering coefficient vs. `S` — clustering does not track `S` here.
#
# The analysis reuses the exact Reactome → graph construction from
# `doc/figures/Fig1+Fig2F-biological_networks.ipynb` (cell "extract ringy
# Reactome pathways"), but records every qualifying pathway instead of
# discarding the ones below threshold. Reactome contains nested/aliased pathway
# definitions that yield byte-for-byte the *same* network; we collapse these by
# largest-component identity so each unique network is counted once (and report
# how many pathway definitions map to it).
#
# **Format note.** This is a *percent-format* script (`# %%` cells). It runs
# cell-by-cell in the VS Code Python Interactive Window (same Jupyter extension
# used for notebooks) and as a plain script (`python
# reactome_ringscore_distribution.py`), while staying pure text for clean
# version control.

# %%
import os
import hashlib
import xml.etree.ElementTree as ET
from itertools import product as iter_product
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.stats import spearmanr

import ringity as rty

rty.set_theme()

# CEMM house palette (mirrors src/ringity/utils/plotting/styling.py) so this file has no
# dependency on internal module paths.
CEMM_DARK = (0 / 255, 43 / 255, 50 / 255)
CEMM_1 = (0 / 255, 85 / 255, 100 / 255)
CEMM_2 = (0 / 255, 140 / 255, 160 / 255)
CEMM_3 = (64 / 255, 185 / 255, 212 / 255)
CEMM_4 = (212 / 255, 236 / 255, 242 / 255)

# on-brand sequential colormap (light → dark teal) for encoding a quantitative value
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "cemm_seq", [CEMM_4, CEMM_3, CEMM_2, CEMM_1, CEMM_DARK]
)

# Analysis parameters (match the manuscript / original notebook).
MIN_NODES = 100  # exclude small pathways to avoid size-related artifacts
RING_THRESHOLD = 0.6  # "dominant ring" cutoff used in the manuscript

# SBML level-3 namespace used by the Reactome export.
SBML_NS = "{http://www.sbml.org/sbml/level3/version1/core}"


# %%
# --- Path handling -----------------------------------------------------------------
# Anchor on this file's location, with fallbacks so it also works when the interactive
# CWD differs from the file directory.
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:  # interactive cell with no __file__
    BASE_DIR = Path.cwd()


def _find_data_dir():
    candidates = [
        BASE_DIR / ".." / "data",
        Path.cwd() / "doc" / "data",
        Path.cwd() / ".." / "data",
        Path.cwd() / "data",
    ]
    for cand in candidates:
        cand = cand.resolve()
        if (cand / "raw_data" / "immune_network").exists():
            return cand
    return (BASE_DIR / ".." / "data").resolve()


DATA_DIR = _find_data_dir()
ROOT_DIR = DATA_DIR / "raw_data" / "immune_network" / "homo_sapiens"
RESULTS_DIR = DATA_DIR / "results"
CACHE_FILE = RESULTS_DIR / "reactome_ringscores.csv"
FIG_STEM = BASE_DIR / "reactome_ringscore_distribution"

if not ROOT_DIR.exists():
    raise FileNotFoundError(
        f"Reactome SBML files not found at {ROOT_DIR}.\n"
        "Run the 'Reactome networks → Download and filter' cells of "
        "doc/figures/Fig1+Fig2F-biological_networks.ipynb first; they download and "
        "extract homo_sapiens.3.1.sbml.tgz from reactome.org."
    )

print(f"DATA_DIR:  {DATA_DIR}")
print(f"ROOT_DIR:  {ROOT_DIR}  ({len(list(ROOT_DIR.glob('*.sbml')))} pathway files)")


# %% [markdown]
# ## Build one graph per Reactome pathway
#
# Following the manuscript Methods: each molecule is a node; species annotated as
# *simple chemicals* (`SBO:0000247`) are removed to avoid star structures around common
# small molecules; products are connected to reactants within each reaction; and we keep
# the largest connected component. Pathways whose largest component has fewer than
# `MIN_NODES` nodes are excluded.


# %%
def build_pathway_graph(sbml_path):
    """Construct the reaction graph for one Reactome SBML pathway file."""
    model = ET.parse(sbml_path).getroot().find(SBML_NS + "model")

    small_molecules = {
        species.attrib["id"]
        for species in model.find(SBML_NS + "listOfSpecies")
        if species.attrib.get("sboTerm") == "SBO:0000247"
    }

    G = nx.Graph(name=model.attrib["name"], id=model.attrib["id"])
    for reaction in model.find(SBML_NS + "listOfReactions"):
        products = reaction.find(SBML_NS + "listOfProducts")
        reactants = reaction.find(SBML_NS + "listOfReactants")
        if products is None or reactants is None:
            continue
        prod = {
            p.attrib["species"]
            for p in products
            if p.attrib["species"] not in small_molecules
        }
        reac = {
            r.attrib["species"]
            for r in reactants
            if r.attrib["species"] not in small_molecules
        }
        G.add_edges_from((r, p) for r, p in iter_product(reac, prod))
    return G


def compute_reactome_scores(root_dir, min_nodes=MIN_NODES, use_cache=True):
    """Ring score + basic descriptors for every Reactome pathway with a large component.

    Results are cached to ``CACHE_FILE``; pass ``use_cache=False`` to force recompute.
    """
    if use_cache and CACHE_FILE.exists():
        print(f"Loading cached scores from {CACHE_FILE}")
        return pd.read_csv(CACHE_FILE)

    files = sorted(p for p in os.listdir(root_dir) if p.endswith(".sbml"))
    records = []
    for nr, fname in enumerate(files, 1):
        print(
            f"computing… {100 * nr / len(files):5.1f}%  ({len(records)} kept)", end="\r"
        )
        try:
            G = build_pathway_graph(root_dir / fname)
        except ET.ParseError:
            continue
        if not G.number_of_edges():
            continue

        H = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        if len(H) < min_nodes:
            continue

        # fingerprint of the largest component's node set: identical hashes mark
        # structurally identical (nested/aliased) Reactome pathway definitions.
        lcc_hash = hashlib.md5("|".join(sorted(H.nodes)).encode()).hexdigest()[:10]

        records.append(
            {
                "stable_id": fname[:-5],  # R-HSA-NNNNN (linkable on reactome.org)
                "pathway_id": G.graph["id"],
                "name": G.graph["name"],
                "n_nodes_full": G.number_of_nodes(),
                "n_nodes_lcc": H.number_of_nodes(),
                "n_edges_lcc": H.number_of_edges(),
                "density": nx.density(H),
                "clustering": nx.average_clustering(H),  # mean of local clustering
                "ring_score": rty.pdiagram(H).ring_score(),
                "lcc_hash": lcc_hash,
            }
        )
    print()

    df = pd.DataFrame(records).sort_values("ring_score", ascending=False)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_FILE, index=False)
    print(f"Computed {len(df)} pathways → cached to {CACHE_FILE}")
    return df


def deduplicate(df):
    """Collapse pathways sharing an identical largest connected component.

    Reactome's nested/aliased pathway definitions can yield byte-for-byte the same
    network (e.g. several cell-junction pathways). We keep one representative row per
    unique component — the shortest member name, i.e. the most general — and record how
    many pathway definitions map to it (`n_pathways`) and which (`members`). All numeric
    descriptors are identical within a group by construction.
    """
    reps = []
    for _, group in df.groupby("lcc_hash", sort=False):
        rep = group.loc[group["name"].str.len().idxmin()].copy()
        rep["n_pathways"] = len(group)
        rep["members"] = "; ".join(sorted(group["name"]))
        reps.append(rep)
    nets = pd.DataFrame(reps).sort_values("ring_score", ascending=False)
    return nets.reset_index(drop=True)


# %%
# Compute (or load) the scores. First run iterates all SBML files (~minutes); subsequent
# runs read the CSV instantly. Set use_cache=False to recompute after a Reactome update.
df = compute_reactome_scores(ROOT_DIR, use_cache=True)

# Collapse structurally identical (nested/aliased) pathway definitions to unique networks.
nets = deduplicate(df)
ringy = nets[nets["ring_score"] > RING_THRESHOLD].sort_values(
    "ring_score", ascending=False
)
rest = nets[nets["ring_score"] <= RING_THRESHOLD]
n_defs = int(nets["n_pathways"].sum())
print(
    f"{len(nets)} unique networks (from {n_defs} pathway definitions, N ≥ {MIN_NODES})"
)
print(f"{len(ringy)} unique networks with S > {RING_THRESHOLD}")
multi = nets[nets["n_pathways"] > 1]
print(
    f"{len(multi)} networks map to >1 definition; "
    f"largest collapse = ×{int(nets['n_pathways'].max())}"
)

# How do the basic descriptors relate to S across networks? (cf. manuscript's global
# NDEx claim that clustering is the descriptor most positively associated with S.)
for col in ["n_nodes_lcc", "density", "clustering"]:
    rho, pval = spearmanr(nets[col], nets["ring_score"])
    print(f"  Spearman(S, {col:>11}) = {rho:+.2f}  (p = {pval:.2g})")
print(
    f"  clustering — ringy median {ringy['clustering'].median():.3f} "
    f"vs rest {rest['clustering'].median():.3f} "
    f"(uniformly low; max over all = {nets['clustering'].max():.3f})"
)
ringy[["name", "n_pathways", "n_nodes_lcc", "clustering", "ring_score"]]


# %% [markdown]
# ## The figure
#
# A 2×2 figure:
#
# - **(a)** distribution of `S` across all unique networks, with the `S > 0.6` tail
#   highlighted;
# - **(b)** the networks above threshold, named and ranked (labels carry `×N` when one
#   network corresponds to several Reactome pathway definitions);
# - **(c)** `S` vs. network size — weak rank correlation, so not a size artifact;
# - **(d)** `S` vs. clustering coefficient — `S` on the y-axis as in (c), so the two
#   bottom panels are directly comparable; a flat cloud, i.e. clustering does not track `S`.


# %%
def _annotate(ax, text, xy=(0.04, 0.96), ha="left", va="top"):
    ax.text(
        *xy,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize="large",
        bbox=dict(boxstyle="round,pad=0.4", fc=CEMM_4, ec=CEMM_2, lw=1.5),
    )


def make_figure(nets, ringy, max_named=20):
    n_defs = int(nets["n_pathways"].sum())
    dnorm = LogNorm(vmin=nets["density"].min(), vmax=nets["density"].max())
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], hspace=0.30, wspace=0.45)
    ax_a = fig.add_subplot(gs[0, 0])  # distribution
    ax_b = fig.add_subplot(gs[0, 1])  # named tail
    ax_c = fig.add_subplot(gs[1, 0])  # S vs size
    ax_d = fig.add_subplot(gs[1, 1])  # clustering vs S

    # --- (a) distribution of S ---------------------------------------------------
    bins = np.arange(0.0, 1.0001, 0.05)  # 0.6 falls on a bin edge → clean split
    counts, edges, patches = ax_a.hist(nets["ring_score"], bins=bins, edgecolor="white")
    for patch, left in zip(patches, edges[:-1]):
        patch.set_facecolor(CEMM_1 if left >= RING_THRESHOLD else CEMM_3)
    ax_a.axvline(RING_THRESHOLD, color=CEMM_DARK, ls="--", lw=2)
    ax_a.set_xlim(0, 1)
    ax_a.set_xlabel("ring score $S$")
    ax_a.set_ylabel("number of networks")
    ax_a.set_title(
        "(a)  Distribution across Reactome networks",
        loc="left",
        fontsize="large",
        fontweight="bold",
    )
    _annotate(
        ax_a,
        f"{len(nets)} networks\n(from {n_defs} pathway definitions)\n"
        f"{len(ringy)} with $S$ > {RING_THRESHOLD}",
        xy=(0.96, 0.95),
        ha="right",
    )

    # --- (b) named tail above threshold (labels on the right to avoid leak) -------
    top = ringy.head(max_named).iloc[::-1]  # largest at top of the axis
    y = np.arange(len(top))
    labels = []
    for name, npw in zip(top["name"], top["n_pathways"]):
        lab = name if len(name) <= 42 else name[:39] + "…"
        labels.append(f"{lab}  (×{int(npw)})" if npw > 1 else lab)
    x_right = 1.10
    ax_b.hlines(
        y, RING_THRESHOLD, x_right, color="0.85", lw=0.8, zorder=0
    )  # row guides
    ax_b.hlines(y, RING_THRESHOLD, top["ring_score"], color=CEMM_2, lw=2.5, zorder=1)
    ax_b.scatter(top["ring_score"], y, s=90, color=CEMM_1, zorder=2)
    for yi, s in zip(y, top["ring_score"]):
        ax_b.text(
            s + 0.012,
            yi,
            f"{s:.2f}",
            va="center",
            ha="left",
            fontsize=9,
            color=CEMM_DARK,
            zorder=3,
        )
    ax_b.axvline(RING_THRESHOLD, color=CEMM_DARK, ls="--", lw=2)
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels, fontsize=9)
    ax_b.yaxis.tick_right()  # names extend into the outer-right margin, never into (a)
    ax_b.tick_params(axis="y", length=0)
    ax_b.set_ylim(-0.6, len(top) - 0.4)
    ax_b.set_xlim(RING_THRESHOLD, x_right)
    ax_b.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax_b.set_xlabel("ring score $S$")
    ax_b.set_title(
        "(b)  Networks with a dominant ring ($S$ > 0.6)",
        loc="left",
        fontsize="large",
        fontweight="bold",
    )

    # --- (c) S vs. network size --------------------------------------------------
    rho_c, p_c = spearmanr(nets["n_nodes_lcc"], nets["ring_score"])
    sc = ax_c.scatter(
        nets["n_nodes_lcc"],
        nets["ring_score"],
        s=70,
        c=nets["density"],
        cmap=DENSITY_CMAP,
        norm=dnorm,
        edgecolor=CEMM_DARK,
        linewidth=0.6,
        zorder=2,
    )
    ax_c.scatter(
        ringy["n_nodes_lcc"],
        ringy["ring_score"],
        s=190,
        facecolors="none",
        edgecolor="#c0392b",
        linewidth=2.0,
        zorder=3,
        label="$S$ > 0.6",
    )
    ax_c.axhline(RING_THRESHOLD, color=CEMM_DARK, ls="--", lw=2)
    ax_c.set_xscale("log")
    ax_c.set_xlabel("number of nodes (largest connected component)")
    ax_c.set_ylabel("ring score $S$")
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.set_title(
        "(c)  Ring score vs. network size",
        loc="left",
        fontsize="large",
        fontweight="bold",
    )
    ax_c.legend(loc="upper right", fontsize="large", frameon=True)
    _annotate(ax_c, f"Spearman $\\rho$ = {rho_c:.2f}  (p = {p_c:.2g})")

    # --- (d) S vs. clustering (axes matched to panel c: S on y) ------------------
    rho_d, p_d = spearmanr(nets["clustering"], nets["ring_score"])
    ax_d.scatter(
        nets["clustering"],
        nets["ring_score"],
        s=70,
        c=nets["density"],
        cmap=DENSITY_CMAP,
        norm=dnorm,
        edgecolor=CEMM_DARK,
        linewidth=0.6,
        zorder=2,
    )
    ax_d.scatter(
        ringy["clustering"],
        ringy["ring_score"],
        s=190,
        facecolors="none",
        edgecolor="#c0392b",
        linewidth=2.0,
        zorder=3,
        label="$S$ > 0.6",
    )
    ax_d.axhline(RING_THRESHOLD, color=CEMM_DARK, ls="--", lw=2)
    ax_d.set_ylim(-0.02, 1.02)
    ax_d.set_xlabel("clustering coefficient\n(mean of local clustering)")
    ax_d.set_ylabel("ring score $S$")
    ax_d.set_title(
        "(d)  Ring score vs. clustering",
        loc="left",
        fontsize="large",
        fontweight="bold",
    )
    ax_d.legend(loc="upper right", fontsize="large", frameon=True)
    # ringy points sit top-left here; annotate the empty lower-right
    _annotate(
        ax_d,
        f"Spearman $\\rho$ = {rho_d:.2f}  (p = {p_d:.2g})",
        xy=(0.96, 0.04),
        ha="right",
        va="bottom",
    )

    # --- shared density colour scale, centred in the gap between (c) and (d) ------
    fig.canvas.draw()  # finalise axes positions before reading them
    pc, pd_ = ax_c.get_position(), ax_d.get_position()
    cbar_w, cbar_h = 0.013, pc.height * 0.72
    cax = fig.add_axes(
        [
            (pc.x1 + pd_.x0) / 2 - cbar_w / 2,
            pc.y0 + (pc.height - cbar_h) / 2,
            cbar_w,
            cbar_h,
        ]
    )
    cbar = fig.colorbar(sc, cax=cax)
    cax.yaxis.set_ticks_position("left")  # ticks/label face panel (c)'s empty margin
    cax.yaxis.set_label_position("left")
    cbar.set_ticks([1e-3, 1e-2, 1e-1])
    cbar.set_ticklabels(["0.001", "0.01", "0.1"])
    cbar.set_label("network density", fontsize="large")
    cbar.ax.tick_params(labelsize="large")
    cbar.outline.set_linewidth(1.0)

    fig.suptitle(
        "Ring structure across Reactome pathways",
        fontsize="xx-large",
        fontweight="bold",
        y=0.99,
    )
    return fig


fig = make_figure(nets, ringy)
fig

# %%
# Save the figure next to this script (transparent, like the manuscript figures).
fig.savefig(FIG_STEM.with_suffix(".pdf"), bbox_inches="tight", transparent=True)
fig.savefig(FIG_STEM.with_suffix(".png"), bbox_inches="tight", dpi=200)
print(
    f"Saved {FIG_STEM.with_suffix('.pdf').name} and {FIG_STEM.with_suffix('.png').name}"
)

# %% [markdown]
# ### Reading the figure for the rebuttal
#
# - Panel (a): a dominant ring (`S > 0.6`) is the *exception*, not the rule — most networks
#   have low/intermediate `S`. This is the requested distribution.
# - Panel (b): names the above-threshold networks, so the claim is concrete and checkable.
#   Where one network is labelled `×N`, it is shared by `N` nested/aliased Reactome pathway
#   definitions (the top `S ≈ 0.92` network corresponds to four cell-junction pathways).
# - Panel (c): the rank correlation between `S` and size is weak (ρ ≈ −0.1, n.s.), so the
#   ring signal is not explained by how many nodes a network has.
# - Panel (d): clustering does **not** track `S` — a flat cloud (ρ ≈ −0.1, n.s.). Clustering
#   is uniformly tiny across all Reactome networks (median ≈ 0.017, max ≈ 0.15; these
#   reaction graphs are nearly tree-/chain-like), and the *highest*-clustering networks
#   (Translation, GPCR signalling, DNA-repair) are the *low*-`S`, non-ringy ones. This
#   contrasts with the manuscript's global NDEx result that clustering is the descriptor most
#   positively associated with `S` (R² = 0.30): that association does not hold within this
#   homogeneous set, so the Reactome ring signal is not a clustering artifact either.
#
# The per-definition table `reactome_ringscores.csv` (id, name, sizes, density, clustering,
# `S`, `lcc_hash`) is the backing data; `nets` is the deduplicated per-network frame with
# `n_pathways` and `members` for any further per-pathway discussion in the response letter.
