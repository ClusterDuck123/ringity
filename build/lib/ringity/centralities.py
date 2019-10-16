#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 15:26:03 2018

@author: myoussef
"""
import time
import scipy.sparse
import scipy.stats
import numpy as np
import networkx as nx


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


def current_distance(G, verbose=False):
    
    if nx.number_of_selfloops(G) > 0:
            if verbose:
                print('Self loops in graph detected. They will be removed!')
            G.remove_edges_from(G.selfloop_edges())
    
    
    N = G.number_of_nodes()
    A = nx.adjacency_matrix(G)
    
    t1 = time.time()
    F = current_flow_matrix(A)
    t2 = time.time()
    
    if verbose:
        print(f'Time for current_flow_matrix calculation: {t2-t1}sec')
    
    edges      = edge_extractor(A)
    edge_dict  = {}

    t1 = time.time()
    for ei, e in enumerate(edges):
        F_ei = F[ei,:]
        rank = scipy.stats.rankdata(F_ei)
        edge_dict[e] = sum([(2*rank[j]-1-N)*F[ei,j] for j in range(N)])
    t2 = time.time()
    
    if verbose:
        print(f'Time for current_distance loop: {t2-t1}sec')
    
    return edge_dict



# ----------------------------- ONLY FOR TESTING -----------------------------

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


def newman_measure(G):
    N = len(G)
    T = prepotential(G)
    I = np.zeros(N)
    
    for s in range(N):
        for t in range(s+1,N):
            I += np.array([
                    sum([abs(T[i,s]-T[i,t]-T[j,s]+T[j,t]) for j in G[i]]) 
                                                            for i in range(N)])
    return (I+(N-1)) / (N*(N-1))


