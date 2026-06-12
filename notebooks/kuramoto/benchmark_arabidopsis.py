"""Sparse Kuramoto derivative for the Arabidopsis gene network.

Dense meshgrid approach: O(N^2) memory and compute.
Sparse approach: O(nnz) -- only touches the ~1M nonzero entries.
"""

import time
import numpy as np
import networkx as nx
import scipy.sparse as sp
from pathlib import Path
from kuramoto import Kuramoto
from scipy.integrate import odeint

GENE_GML = Path(__file__).parent.parent.parent / "data" / "empirical_networks" / "gene.gml"
DT = 0.001
T_FULL = 1000.0


def load_network():
    print("Loading Arabidopsis gene network...")
    G = nx.read_gml(GENE_GML)
    G.remove_edges_from(nx.selfloop_edges(G))
    adj_sparse = nx.adjacency_matrix(G).astype(float)  # scipy CSR
    adj_dense = adj_sparse.toarray()
    n, e = G.number_of_nodes(), G.number_of_edges()
    print(f"  nodes={n}, edges={e}, density={2*e / (n*(n-1)):.4f}")
    print(f"  dense adj matrix: {adj_dense.nbytes / 1e6:.0f} MB")
    print(f"  sparse nnz: {adj_sparse.nnz:,} ({adj_sparse.nnz / adj_dense.size * 100:.1f}% fill)")
    return adj_dense, adj_sparse


def sparse_derivative(angles_vec, natfreqs, coupling_vec, adj_coo):
    """Kuramoto derivative using the COO nonzeros only -- O(nnz) instead of O(N^2).

    Computes: dxdt[i] = natfreqs[i] + coupling[i] * sum_j( A[j,i] * sin(angles[j] - angles[i]) )
    """
    sin_diffs = np.sin(angles_vec[adj_coo.row] - angles_vec[adj_coo.col])
    contributions = adj_coo.data * sin_diffs
    interactions = np.bincount(adj_coo.col, weights=contributions, minlength=len(natfreqs))
    return natfreqs + coupling_vec * interactions


def make_state(n, seed=42):
    rng = np.random.default_rng(seed)
    natfreqs = 1 + 0.15 * rng.standard_normal(n)
    angles_vec = rng.uniform(0, 2 * np.pi, n)
    return natfreqs, angles_vec


def verify(adj_dense, adj_sparse):
    print("\nVerifying sparse == dense for a single derivative call...")
    n = adj_dense.shape[0]
    natfreqs, angles_vec = make_state(n)

    model = Kuramoto(coupling=1.0, dt=DT, T=DT, n_nodes=n, natfreqs=natfreqs)
    n_interactions = (adj_dense != 0).sum(axis=0)
    coupling_vec = model.coupling / n_interactions

    dxdt_dense = model.derivative(angles_vec, 0, adj_dense, coupling_vec)

    adj_coo = adj_sparse.tocoo()
    dxdt_sparse = sparse_derivative(angles_vec, natfreqs, coupling_vec, adj_coo)

    max_err = np.max(np.abs(dxdt_dense - dxdt_sparse))
    print(f"  max absolute error between implementations: {max_err:.2e}")
    assert max_err < 1e-10, f"Mismatch: max error = {max_err}"
    print("  OK: results agree to within floating-point precision")


def benchmark_dense(adj_dense):
    n = adj_dense.shape[0]
    natfreqs, angles_vec = make_state(n)

    model = Kuramoto(coupling=1.0, dt=DT, T=DT, n_nodes=n, natfreqs=natfreqs)
    n_interactions = (adj_dense != 0).sum(axis=0)
    coupling_vec = model.coupling / n_interactions

    model.derivative(angles_vec, 0, adj_dense, coupling_vec)  # warm-up

    n_reps = 3
    t0 = time.perf_counter()
    for _ in range(n_reps):
        model.derivative(angles_vec, 0, adj_dense, coupling_vec)
    return (time.perf_counter() - t0) / n_reps


def benchmark_sparse(adj_sparse):
    n = adj_sparse.shape[0]
    natfreqs, angles_vec = make_state(n)

    adj_coo = adj_sparse.tocoo()
    # coupling_vec: one-time setup cost, same as in the dense path
    n_interactions = np.bincount(adj_coo.col, minlength=n)
    coupling_vec = 1.0 / n_interactions
    sparse_derivative(angles_vec, natfreqs, coupling_vec, adj_coo)  # warm-up

    n_reps = 10
    t0 = time.perf_counter()
    for _ in range(n_reps):
        sparse_derivative(angles_vec, natfreqs, coupling_vec, adj_coo)
    return (time.perf_counter() - t0) / n_reps


def main():
    adj_dense, adj_sparse = load_network()

    verify(adj_dense, adj_sparse)

    print("\nBenchmarking dense derivative (original)...")
    t_dense = benchmark_dense(adj_dense)
    print(f"  {t_dense:.3f} s per call")

    print("\nBenchmarking sparse derivative...")
    t_sparse = benchmark_sparse(adj_sparse)
    print(f"  {t_sparse:.4f} s per call")

    print(f"\nSpeedup: {t_dense / t_sparse:.0f}x")

    n_steps = int(T_FULL / DT)
    print(f"\nExtrapolated full run ({n_steps:,} steps):")
    print(f"  dense:  {t_dense  * n_steps / 3600:.1f} h")
    print(f"  sparse: {t_sparse * n_steps / 3600:.1f} h")


if __name__ == "__main__":
    main()
