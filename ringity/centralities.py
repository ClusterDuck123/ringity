from ringity.exceptions import DisconnectedGraphError
from scipy.stats import rankdata

import time
import scipy.sparse
import scipy.stats
import numpy as np
import networkx as nx

# Newest version
def net_flow(G, efficiency='speed'):
    if not nx.is_connected(G):
        raise DisconnectedGraphError

    L = nx.laplacian_matrix(G, weight=None).toarray()
    C = np.zeros(L.shape)
    C[1:,1:] = np.linalg.inv(L[1:,1:])

    N = G.number_of_nodes()
    E = G.number_of_edges()
    B = nx.incidence_matrix(G, oriented=True).T #shape=(nodes,edges)

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


def resistance(G):
    L = nx.laplacian_matrix(G)
    Gamm = np.linalg.pinv(L.A)
    diag = np.diag(Gamm)
    return (-2*Gamm + diag).T + diag

# ----------------------------- ONLY FOR TESTING -----------------------------

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
