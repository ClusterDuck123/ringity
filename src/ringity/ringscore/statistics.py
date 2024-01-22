import numpy as np
import scipy.stats as ss


def pvalues_from_pdiagram(pdgm, remove_top_n=0, model="gumbel"):
    pratios = pdgm.pratios
    fit_pratios = remove_n_largest(pratios, remove_top_n)
    if model == "pareto":
        alpha = estimate_tail_index(fit_pratios)
        pvalues = 1 - ss.pareto(b=alpha).cdf(pratios)
    elif model == "gumbel":
        lvalues = l_transform(pratios, fit_pratios)
        pvalues = np.exp(-np.exp(lvalues))
    return pvalues


def remove_n_largest(pratios, n):
    temp = np.argpartition(-pratios, n)
    result_args = temp[:n]
    return np.delete(pratios, result_args)


def estimate_tail_index(pratios):
    alpha = len(pratios) / sum(np.log(pratios))
    return alpha


def l_transform(pratios, fit_pratios = None):
    if fit_pratios is None:
        fit_pratios = pratios
    loglog_ratio = np.log(np.log(pratios))
    corr_factor = np.euler_gamma + np.mean(np.log(np.log(fit_pratios)))
    lvalues = loglog_ratio - corr_factor
    return lvalues


def nb_of_cycles(pdgm):
    pvalues = pvalues_from_pdiagram(pdgm)
    nb_cycles = len(pvalues[pvalues < 0.05 / len(pvalues)])
    return nb_cycles


def min_pvalue_from_pdiagram(
    pdgm,
    bonferroni=True,
    remove_top_n=0,
    model="gumbel",
):
    pvalues = pvalues_from_pdiagram(
        pdgm,
        remove_top_n=remove_top_n,
        model=model,
    )
    min_pvalue = min(pvalues)
    if bonferroni:
        min_pvalue = min_pvalue * len(pvalues)
    return min_pvalue
