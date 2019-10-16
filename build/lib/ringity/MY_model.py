#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:22:49 2018

@author: myoussef
"""

from numpy import pi
from itertools import combinations
from scipy.stats import rv_continuous

import numpy as np
import networkx as nx

# ------------------------ INTERACTION STRENGTH LAWS -------------------------

def exponential(delta, mu):
    return np.exp(-mu*delta)

def power_law(delta, gamma):
    return float(delta/pi+1)**(-gamma)

def linear(delta, max_delta):
    return max_delta - delta


# -------------------- INTERACTION STRENGTH DISTRIBUTIONS --------------------    
    
def exp_pdf(z, mu=1, alpha=0.5):
    A = max(z-alpha, (1-alpha)*np.exp(-mu))
    B = min(1-alpha, z)
    integral = lambda t : np.log(t)/(mu*alpha)
    
    if A >= B:
        return 0
    else:
        return integral(B) - integral(A)
       
def pdf_pow(z, gamma=1, alpha=0.5):
    A = max(0    , z+alpha-1)
    B = min(alpha, z-(1-alpha)/2**gamma)
    integral = lambda t : ((1-alpha)/(z-t))**(1/gamma) / alpha
    
    if A >= B:
        return 0
    else:
        return integral(B) - integral(A)

    
class MyExpRV(rv_continuous):
    "Edge weight distribution for exponential interaction strength."
    def _pdf(self, z, mu, alpha):
        return exp_pdf(z, mu=mu, alpha=alpha)
    
class MyPowRV(rv_continuous):
    "Edge weight distribution for power law interaction strength."
    def _pdf(self, z, mu, alpha):
        return exp_pdf(z, mu=mu, alpha=alpha)



# ----------------------------- GETTER FUNCTIONS -----------------------------
        
def _get_interaction_law_function(law):
    if   law == 'exp':
        return exponential
    elif law == 'pow':
        return power_law
    elif law == 'lin':
        # -------------- NOT IMPLEMENTED YET !!! --------------
        print('Not implemented yet!')
        raise Exception
    else:
        # -------------- NOT IMPLEMENTED YET !!! --------------
        print('Not implemented yet!')
        raise Exception
        
def _get_empty_graph(N, time_dist):
    G = nx.Graph()
    G.add_nodes_from(range(N))
    times = 2*pi*time_dist(size=N)
    nx.set_node_attributes(G, 
                           values={n:times[n] for n in range(N)}, 
                           name='time')
    return G, times
        

        
# -------------------------- MENCHE-YOUSSEF MODELS ---------------------------    
    
def MY_full_continuous_model(N, alpha=0.5, shape=1, law='exp', 
                             time_dist=np.random.uniform):
    law = _get_interaction_law_function(law)
    G, times = _get_empty_graph(N, time_dist)

    for (u,v) in combinations(G.nodes,2): 
        r = np.random.uniform()
        delta = min(abs(times[u]-times[v]),2*pi-abs(times[u]-times[v]))
        weight = alpha*r + (1-alpha)*law(delta, shape)
        G.add_edge(u,v, weight=weight)
    return G


def MY_sparse_continuous_model(N, alpha=0.5, shape=1, p=0.1, law='exp'):
    G = MY_full_continuous_model(N, shape=shape, alpha=alpha, law=law)
    th = sorted(G[v][w]['weight'] for (v,w) in G.edges)[int((1-p)*N*(N-1)/2)]
    for (u,v) in combinations(G.nodes,2):
        if G[u][v]['weight'] < th:
            G.remove_edge(u,v)
        else:
            G[u][v].pop('weight')
    return G
    
    
    
# ------------------------ DEPRICATED STUFF ----------------------------------
    
def MY_discrete_model(N, T, gamma=1, dist='exp'):
    if dist == 'exp':
        dist = exponential
    elif dist == 'pow':
        dist = power_law
    elif dist == 'lin':
        gamma = np.floor(T/2)+1
        dist = linear
    else:
        raise Exception
        
    S = sum([dist(delta, gamma) for delta in range(int(np.floor(T/2)))])
    
    G = nx.Graph()
    G.add_nodes_from(range(N))
    mod = np.random.randint(T, size=N)
    nx.set_node_attributes(G, 
                           values={n:mod[n] for n in range(N)}, 
                           name='mod')

    for i in range(N): 
        for j in range(i+1,N):
            r = np.random.uniform(0,S)
            delta = min((mod[i]-mod[j])%T,(mod[j]-mod[i])%T)
            if r < dist(delta, gamma):
                G.add_edge(i,j)
    return G