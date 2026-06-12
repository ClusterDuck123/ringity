"""Run sparse Kuramoto on the Arabidopsis circadian gene network.

Usage:
    python run_arabidopsis_sparse.py --n_steps 100 --dt 1.0 \\
        --genes_file circadian_genes.json --output arabidopsis_angles.json
    python run_arabidopsis_sparse.py --euler   # fixed-step Euler instead of LSODA

The genes file is a JSON with structure:
    { "morning": [{"symbol": "CCA1", "agi": "AT2G46830"}, ...], ... }

The output JSON contains the angle timeseries (and actual time axis) for
those focal genes, ready for plot_gene_angles.py.

Gene phase assignments follow the PRR-wave model:
    Greenham & McClung (2015) Nat Rev Genet 16:598-610
"""

import argparse
import json
import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.integrate import odeint
from pathlib import Path

GENE_GML = Path(__file__).parent.parent.parent / "data" / "empirical_networks" / "gene.gml"
SEED     = 42

# Full partition used for the coherence summary printout.
# Genes absent from the network (PRR9, LUX, RVE6, SRR1) are omitted.
# Ref: Greenham & McClung 2015; individual primary refs inline.
CIRCADIAN_PHASES = {
    "morning": {                          # ZT0–6
        "CCA1": "AT2G46830",             # peak ZT0–2
        "LHY":  "AT1G01060",             # peak ZT0–2
    },
    "day": {                              # ZT6–12
        "PRR7":  "AT5G02810",            # peak ZT4–8
        "PRR5":  "AT5G24470",            # peak ZT8–10
        "PRR3":  "AT5G60100",            # peak ZT10–12
        "GI":    "AT1G22770",            # peak ZT10–14
        "RVE8":  "AT3G09600",            # mRNA ZT0; protein/EE-binding activity ZT3–8
        "RVE4":  "AT5G02840",            # mRNA ZT0; protein/EE-binding activity ZT3–8
        "LNK1":  "AT5G64170",            # peak ZT8–10
        "LNK2":  "AT3G54500",            # peak ZT8–10
        "FKF1":  "AT1G68050",            # peak ZT9–12
    },
    "evening": {                          # ZT12–18
        "TOC1": "AT5G61380",             # peak ZT12–14
        "ELF3": "AT2G25930",             # peak ZT12–16
        "ELF4": "AT2G40080",             # peak ZT14–16
        "CHE":  "AT5G08330",             # peak ZT14–18
    },
    "night": {                            # ZT18–24
        "ZTL":  "AT5G57360",             # protein peak ZT16–20
        "LKP2": "AT2G18915",             # protein peak ZT18–24
    },
}


# ---------------------------------------------------------------------------
# Sparse derivative  (O(nnz), not O(N^2))
# ---------------------------------------------------------------------------

def sparse_derivative(angles_vec, t, natfreqs, coupling_vec, adj_coo):
    sin_diffs = np.sin(angles_vec[adj_coo.row] - angles_vec[adj_coo.col])
    contributions = adj_coo.data * sin_diffs
    interactions = np.bincount(adj_coo.col, weights=contributions,
                               minlength=len(natfreqs))
    return natfreqs + coupling_vec * interactions


# ---------------------------------------------------------------------------
# Integrators
# ---------------------------------------------------------------------------

def integrate_lsoda(angles_vec, natfreqs, coupling_vec, adj_coo, n_steps, dt):
    t = np.linspace(0, n_steps * dt, n_steps + 1)
    return odeint(sparse_derivative, angles_vec, t,
                  args=(natfreqs, coupling_vec, adj_coo)), t


def integrate_euler(angles_vec, natfreqs, coupling_vec, adj_coo, n_steps, dt):
    n = len(angles_vec)
    activity = np.empty((n_steps + 1, n))
    activity[0] = angles_vec
    for i in range(n_steps):
        activity[i + 1] = activity[i] + dt * sparse_derivative(
            activity[i], None, natfreqs, coupling_vec, adj_coo
        )
    t = np.linspace(0, n_steps * dt, n_steps + 1)
    return activity, t


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_network():
    G = nx.read_gml(GENE_GML)
    G.remove_edges_from(nx.selfloop_edges(G))
    nodelist   = list(G.nodes())
    node_index = {label: i for i, label in enumerate(nodelist)}
    adj_sparse = nx.adjacency_matrix(G, nodelist=nodelist).astype(float)
    adj_coo    = sp.coo_array(adj_sparse)
    return node_index, adj_coo


def load_focal_genes(genes_file, node_index):
    """Load gene-phase assignments from JSON; return only genes present in network."""
    with open(genes_file) as f:
        gene_classes = json.load(f)

    focal = {}  # symbol -> {"agi": ..., "phase": ..., "idx": ...}
    for phase, entries in gene_classes.items():
        for entry in entries:
            symbol, agi = entry["symbol"], entry["agi"]
            if agi in node_index:
                focal[symbol] = {"agi": agi, "phase": phase, "idx": node_index[agi]}
            else:
                print(f"  Warning: {symbol} ({agi}) not found in network — skipping")
    return focal


def build_phase_indices(node_index):
    phase_indices, missing = {}, {}
    for phase, genes in CIRCADIAN_PHASES.items():
        found, absent = {}, []
        for sym, agi in genes.items():
            if agi in node_index:
                found[sym] = node_index[agi]
            else:
                absent.append(sym)
        phase_indices[phase] = found
        if absent:
            missing[phase] = absent
    return phase_indices, missing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(n_steps, dt, genes_file, output, euler):
    print("Loading network...")
    node_index, adj_coo = load_network()
    n = adj_coo.shape[0]
    print(f"  {n} nodes, {adj_coo.nnz} nonzeros")

    focal_genes = load_focal_genes(genes_file, node_index)
    print(f"\nFocal genes ({len(focal_genes)} found in network):")
    for sym, info in focal_genes.items():
        print(f"  {info['phase']:8s}  {sym:6s}  {info['agi']}")

    rng          = np.random.default_rng(SEED)
    natfreqs     = np.ones(n)                       # uniform frequency = 1
    angles_vec   = rng.uniform(0, 2 * np.pi, n)    # random initial phases

    adj_coo_coo    = sp.coo_array(adj_coo)
    n_interactions = np.bincount(adj_coo_coo.col, minlength=n)
    coupling_vec   = 1.0 / n_interactions

    integrator = "Euler" if euler else "LSODA"
    print(f"\nRunning {n_steps} steps (T={n_steps * dt:.4g}, dt={dt}, {integrator})...")
    if euler:
        activity, t = integrate_euler(angles_vec, natfreqs, coupling_vec,
                                      adj_coo_coo, n_steps, dt)
    else:
        activity, t = integrate_lsoda(angles_vec, natfreqs, coupling_vec,
                                      adj_coo_coo, n_steps, dt)
    # activity: (n_steps+1, n_nodes)

    # ---- coherence summary for the full phase partition ----
    phase_indices, missing = build_phase_indices(node_index)
    global_r = np.abs(np.mean(np.exp(1j * activity), axis=1))
    print("\nPhase coherence (t=0 → t_final):")
    print(f"  {'group':10s}  {'r(t=0)':>8s}  {'r(t_end)':>8s}  members")
    for phase, genes in phase_indices.items():
        if not genes:
            continue
        idx  = list(genes.values())
        r0   = float(np.abs(np.mean(np.exp(1j * activity[0,  idx]))))
        rend = float(np.abs(np.mean(np.exp(1j * activity[-1, idx]))))
        print(f"  {phase:10s}  {r0:8.4f}  {rend:8.4f}  {', '.join(genes)}")
    print(f"  {'global':10s}  {float(global_r[0]):8.4f}  {float(global_r[-1]):8.4f}")

    # ---- save focal gene angles ----
    out = {
        "t":        t.tolist(),
        "params":   {"n_steps": n_steps, "dt": dt, "seed": SEED,
                     "integrator": "euler" if euler else "lsoda",
                     "natfreq": 1.0},
        "phase_of": {sym: info["phase"] for sym, info in focal_genes.items()},
        "angles":   {sym: activity[:, info["idx"]].tolist()
                     for sym, info in focal_genes.items()},
    }
    with open(output, "w") as f:
        json.dump(out, f)
    print(f"\nAngle timeseries saved to {output}")


if __name__ == "__main__":
    here = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_steps",    type=int,   default=100)
    parser.add_argument("--dt",         type=float, default=1.0)
    parser.add_argument("--genes_file", type=str,
                        default=str(here / "circadian_genes.json"))
    parser.add_argument("--output",     type=str,
                        default=str(here / "arabidopsis_angles.json"))
    parser.add_argument("--euler",      action="store_true",
                        help="Use fixed-step Euler instead of LSODA")
    args = parser.parse_args()
    main(args.n_steps, args.dt, args.genes_file, args.output, args.euler)
