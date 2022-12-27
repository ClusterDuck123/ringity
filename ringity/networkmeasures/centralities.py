from ringity.classes.exceptions import DisconnectedGraphError
from scipy.stats import rankdata
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

# -------------------- ONLY FOR LEGACY AND TESTING --------------------

def edge_extractor(A):
    N = A.shape[0]
    indices = A.indices
    indptr  = A.indptr

    for i in range(N):
        for index in range(indptr[i], indptr[i+1]):
            j = indices[index]
            if j<i:
                continue
            yield i,j

def laplace(A):
    n, m = A.shape
    diags = A.sum(axis=1)
    D = scipy.sparse.spdiags(diags.flatten(), [0], m, n)
    return D - A

def oriented_incidence_matrix(A):
    assert type(A) == scipy.sparse.csr.csr_matrix
    N = A.shape[0]
    E = int(A.nnz/2)
    B = scipy.sparse.lil_matrix((E, N))

    edges = edge_extractor(A)

    for ei, (u,v) in enumerate(edges):
        B[ei, u] = -1
        B[ei, v] = 1
    return B

def current_flow_matrix(A):
    N = A.shape[0]
    L = laplace(A)

    L_tild = L[1:,1:].toarray()
    T_tild = np.linalg.inv(L_tild)
    C = np.zeros([N,N])
    C[1:,1:] = T_tild

    B = oriented_incidence_matrix(A)

    return B@C

def prepotential(G):
    """
    Returns a sparse csc matrix that multiplied by a supply yields the corresponding
    (absolute) potential.
    """
    L = nx.laplacian_matrix(G).toarray()
    L_tild = L[1:,1:]
    T_tild = np.linalg.inv(L_tild)
    T = np.zeros(L.shape)
    T[1:,1:] = T_tild
    return T

def slow_current_distance(G):
    N = G.number_of_nodes()
    E = G.number_of_edges()
    A = nx.adjacency_matrix(G)
    F = current_flow_matrix(A)
    edge_array = np.zeros(E, dtype=float)

    for s in range(N):
        for t in range(s+1,N):
            edge_array += np.abs(F[:,s] - F[:,t])

    edge_dict = {e:edge_array[ei] for (ei,e) in enumerate(G.edges)}
    return edge_dict

def stupid_current_distance(G):
    """
    Returns a dictionary corresponding to the random walk based edge
    centrality measure.
    """

    T = prepotential(G)
    edge_dict = {e:0 for e in G.edges}
    N = G.number_of_nodes()

    for s in range(N):
        for t in range(s+1,N):
            p = T[:,s] - T[:,t]
            for (v,w) in edge_dict:
                edge_dict[(v,w)] += abs(p[v]-p[w])
    return edge_dict

def current_flow_betweenness(G):
    N = len(G)
    T = prepotential(G)
    I = np.zeros(N)

    for s in range(N):
        for t in range(s+1,N):
            I += np.array([
                    sum([abs(T[i,s]-T[i,t]-T[j,s]+T[j,t]) for j in G[i]])
                                                            for i in range(N)])
    return (I+(N-1)) / (N*(N-1))


def net_flow_old(G, efficiency='speed'):
    if not nx.is_connected(G):
        raise DisconnectedGraphError

    L = nx.laplacian_matrix(G, weight=None).toarray()
    C = np.zeros(L.shape)
    C[1:,1:] = np.linalg.inv(L[1:,1:])

    N = G.number_of_nodes()
    E = G.number_of_edges()
    B = nx.incidence_matrix(G, oriented=True).T # shape -> (nodes,edges)

    if   efficiency == 'memory':
        values = np.zeros(G.number_of_edges())
        for idx, B_row in enumerate(B):
            F_row = B_row@C
            rank = rankdata(F_row)
            values[idx] = np.sum((2*rank-1-N)*F_row)
    elif efficiency == 'speed':
        F = B@C
        F_ranks = np.apply_along_axis(rankdata, arr = F, axis = 1)
        values = np.sum((2*F_ranks-1-N)*F, axis=1)
    else:
        raise Exception("Efficiency unknown.")

    edge_dict  = dict(zip(G.edges, values))
    return edge_dict
