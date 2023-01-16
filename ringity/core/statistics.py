import numpy as np
import scipy.stats as ss

def estimate_tail_index(pdgm):
    alpha = len(pdgm) / sum(np.log(pdgm.pratios))
    return alpha

def pvalues_from_pdiagram(pdgm):
    alpha = estimate_tail_index(pdgm)
    pvalues = 1 - ss.pareto(b = alpha).cdf(pdgm.pratios)
    return pvalues

def nb_of_cycles(pdgm):
    pvalues = pvalues_from_pdiagram(pdgm)
    nb_cycles = len(pvalues[pvalues < 0.05 / len(pvalues)])
    return nb_cycles

