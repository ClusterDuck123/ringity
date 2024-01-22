from ringity.utils.exceptions import DisconnectedGraphError
from numba import njit

import scipy.sparse
import scipy.stats
import numpy as np
import networkx as nx

def current_flow(G,
        inplace = False,
        new_weight_name = None,
        verbose = False):
    
    N = len(G)
    edge_dict = net_flow(G, inplace = False)
    edge_dict = {e : v/((N-1)*(N-2)) for e,v in edge_dict.items()}

    if inplace:
        nx.set_edge_attributes(G, 
                            values = edge_dict, 
                            name = new_weight_name)
    else:
        return edge_dict

@njit
def potential_to_current_flow_edge(C, edge):
    """Calculates the current flow for a single edge given the potential
    matrix."""
    u, v = edge
    F_row = C[u]-C[v]
    ranks = np.empty_like(F_row) # calculate ranks using numpy's argsort function
    ranks[np.argsort(F_row)] = np.arange(1, len(F_row)+1)

    return np.sum((2*ranks-1-len(F_row))*F_row)

@njit
def _current_flow_loop(C, row_idx, col_idx):
    """Loops the function ``potential_to_current_flow_edge`` over all
    edges defined via ``row_idx`` and ``col_idx``.

    The purpose of this function is to use numba's decorator ``@njit``.
    """
    return [potential_to_current_flow_edge(C, (u, v))
                    for (u,v) in zip(row_idx, col_idx)
                            if u <= v]

def net_flow(G, 
        inplace = False,
        new_weight_name = None,
        verbose = False):
    """Calculate edge dictionary corresponding to a weighted adjacency matrices
    of absorbing random-walk based centrality measure on graphs.

    This algorithm is based on an algorithm from Brandes and Fleischer. [1]

    Parameters
    ----------
    G : networkx.classes.graph.Graph

    
    Returns
    -------
    edge_dict : dictionary of edge weights


    References
    ----------
    [1] Brandes, Ulrik, and Daniel Fleischer. "Centrality measures based on
    current flow." Annual symposium on theoretical aspects of computer science.
    Springer, Berlin, Heidelberg, 2005.
    """

    if new_weight_name is None:
        new_weight_name = 'net_flow'
        # TO-DO: WHAT HAPPENS WHEN WEIGHT NAME ALREADY EXISTS?

    node_label = dict(enumerate(G.nodes))
    if not nx.is_connected(G):
        raise DisconnectedGraphError

    # The exact sparse representation is irrelevant from a speed
    # perspective, but differs in the implementation below.
    A_coo = nx.to_scipy_sparse_array(G, format = 'coo')

    if not isinstance(A_coo.dtype, float):
        A_coo = A_coo.astype(np.float64)

    # Inverted Laplacian is going to be dense anyways.
    # So there is no harm in working with dense matrices here.
    L = -A_coo.toarray()
    np.fill_diagonal(L, A_coo.sum(axis = 1))

    # Removing first (or any other) row and corresponding column from the
    # Laplacian matrix leads to an invertible matrix if the graph is connected.
    # The invers of this matrix can be interpreted as a 'potential' matrix
    # that leads to voltage vectors by taking differences.
    C = np.zeros(L.shape)
    C[1:,1:] = np.linalg.inv(L[1:,1:])

    # Calculating the current flow directly involves taking all pairwise distances
    # twice: We need to take all pairwise distances for each source-target pair
    # (leading to a voltage vector V_st), and we need to do that for all source-target
    # pairs (leading to the current flow matrix). This is computationally expensive.
    # We can reduce the complexity by realizing that not all values need to be calculated
    # and that some computations can be reused. This idea is based on an algorithm from
    # Brandes and Fleischer.

    # The implementation here is optimized for COO sparse format, converting
    # CSR to COO is cheap.

    row_idx, col_idx = map(np.array, zip(*((i,j)
                        for (i,j) in zip(A_coo.row, A_coo.col) if i<j)))

    edge_values = _current_flow_loop(C, row_idx, col_idx )
    edge_dict = {(node_label[i], node_label[j]) : edge_values
                        for (i,j,edge_values) in zip(row_idx, col_idx, edge_values)}

    if inplace:
        nx.set_edge_attributes(G, 
                            values = edge_dict, 
                            name = new_weight_name)
    else:
        return edge_dict


def resistance(G):
    L = laplace(nx.to_scipy_sparse_array(G))
    Gamm = np.linalg.pinv(L.A, hermitian = True)
    diag = np.diag(Gamm)
    return (-2*Gamm + diag).T + diag

def laplace(A):
    n, m = A.shape
    diags = A.sum(axis=1)
    D = scipy.sparse.spdiags(diags.flatten(), [0], m, n)
    return D - A