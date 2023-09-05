import numpy as np
import networkx as nx

from scipy.sparse import csc_matrix

def triangle_signature(G, weight = None):
    """
    Returns clustering coefficient and community coefficient.
    This function will ignore self-loops.
    """
    A = nx.to_scipy_sparse_matrix(G, format='lil', weight = weight)
    A.setdiag(0)
    A = csc_matrix(A)
    A.eliminate_zeros()
    A2 = A@A
    o0 = A.sum(axis=1).A
    numer = A.multiply(A2).sum(axis=1).A

    D_denom = o0*o0 - o0
    A_denom = A @o0 - o0

    C_D = np.true_divide(numer, D_denom,
                         out   = np.nan*np.empty_like(numer, dtype=float),
                         where = D_denom!=0).flatten()
    C_A = np.true_divide(numer, A_denom,
                         out   = np.nan*np.empty_like(numer, dtype=float),
                         where = A_denom!=0).flatten()
    return C_D, C_A
    

def clustering_coefficient(G, weight = None):
    """
    Returns clustering coefficient.
    This function will ignore self-loops.
    """
    A = nx.to_scipy_sparse_array(G, format='lil', weight = weight)
    A.setdiag(0)
    A = csc_matrix(A)
    A.eliminate_zeros()
    A2 = A@A
    o0 = A.sum(axis=1).A
    numer = A.multiply(A2).sum(axis=1).A

    denom = o0*o0 - o0

    C = np.true_divide(numer, denom,
                       out   = np.nan*np.empty_like(numer, dtype=float),
                       where = denom!=0).flatten()
    return C    