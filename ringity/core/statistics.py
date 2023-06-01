import numpy as np
import scipy.stats as ss

from statsmodels.stats.multitest import fdrcorrection

def pvalues_from_pdiagram(pdgm, model='gumbel', nb_outliers = 0):
    if model == 'gumbel':
        l_values = l_transform(pdgm, nb_outliers)
        pvals = np.exp(-np.exp(l_values))
    elif model == 'pareto':
        alpha = estimate_tail_index(pdgm, nb_outliers)
        pvals = 1 - ss.pareto(b = alpha).cdf(pdgm.pratios)
    return pvals

def remove_outliers(data, nb_outliers = 0):
    data_part = np.partition(-data, nb_outliers)
    data_nooutlier = -data_part[nb_outliers:]
    return data_nooutlier

def estimate_tail_index(pdgm, nb_outliers = 0):
    pratios = remove_outliers(pdgm.pratios, nb_outliers)
    alpha = len(pratios) / sum(np.log(pratios))
    return alpha

def l_transform(pdgm, nb_outliers = 0):
    pratios = pdgm.pratios
    pratios_nooutlier = remove_outliers(pratios, nb_outliers)
    
    loglog_ratios = np.log(np.log(pratios))
    
    # calculate correction factor
    loglog_mean = np.mean(np.log(np.log(pratios_nooutlier)))
    corr_factor = np.euler_gamma + np.mean(loglog_mean)
    
    l_values = loglog_ratios - corr_factor
    return l_values

## -------------------- NEEDS MODIFICATION --------------------

def get_min_adj_pval(pdgm, nb_outliers = 0, adj_method = 'fdr'):
    l_values = l_transform(pdgm, nb_outliers)
    raw_pvals = np.exp(-np.exp(l_values))
    if adj_method == 'fdr':
        min_adj_pval = min(fdrcorrection(raw_pvals)[1])
    elif adj_method == 'bonferroni':
        min_adj_pval = min(raw_pvals * len(l_values))
    return min_adj_pval

def nb_of_cycles(pdgm):
    pvalues = pvalues_from_pdiagram(pdgm)
    nb_cycles = len(pvalues[pvalues < 0.05 / len(pvalues)])
    return nb_cycles