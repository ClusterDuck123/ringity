import numpy as np

from scipy.integrate import dblquad
from ringity.networkmodel import transformations as rtrafos
from ringity.networkmodel import distributions as rdists

# Default integration precision parameters
DEFAULT_EPSABS = 1.49e-07
DEFAULT_EPSREL = 1.49e-07

#############################################################################
# ---------------------------- RAW INTEGRAL ---------------------------------
#############################################################################


def _local_clustering_integrand_classic(theta_k, theta_j, theta_i, r, c, beta):
    assert theta_j >= 0, (theta_j, theta_k)
    assert theta_k >= 0, (theta_j, theta_k)

    d_ij = rtrafos.angular_distance(theta_i, theta_j)
    d_ik = rtrafos.angular_distance(theta_i, theta_k)
    d_jk = rtrafos.angular_distance(theta_j, theta_k)

    p_ij = rtrafos.interaction_probability(d_ij, r=r, c=c)
    p_ik = rtrafos.interaction_probability(d_ik, r=r, c=c)
    p_jk = rtrafos.interaction_probability(d_jk, r=r, c=c)

    f_j = rdists.pdf_delay(theta_j, beta=beta)
    f_k = rdists.pdf_delay(theta_k, beta=beta)

    return p_ij * p_ik * p_jk * f_j * f_k


def _local_clustering_integrand_noconstants(theta_k, theta_j, theta_i, r, rate):
    assert theta_j >= 0, (theta_j, theta_k)
    assert theta_k >= 0, (theta_j, theta_k)

    d_ij = rtrafos.angular_distance(theta_i, theta_j)
    d_ik = rtrafos.angular_distance(theta_i, theta_k)
    d_jk = rtrafos.angular_distance(theta_j, theta_k)

    s_ij = max(1 - d_ij / (2 * np.pi * r), 0)
    s_ik = max(1 - d_ik / (2 * np.pi * r), 0)
    s_jk = max(1 - d_jk / (2 * np.pi * r), 0)

    e_j = np.exp(-rate * theta_j)
    e_k = np.exp(-rate * theta_k)

    return s_ij * s_ik * s_jk * e_j * e_k


def _local_clustering_integral_classic(
    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
):
    I = dblquad(
        _local_clustering_integrand_classic,
        0,
        2 * np.pi,
        0,
        2 * np.pi,
        args=(theta_i, r, c, beta),
        epsabs=epsabs,
        epsrel=epsrel,
    )[0]
    return I


def _local_clustering_integral_noconstants(
    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
):
    rate = rtrafos.beta_to_rate(beta)

    I = dblquad(
        _local_clustering_integrand_noconstants,
        0,
        2 * np.pi,
        0,
        2 * np.pi,
        args=(theta_i, r, rate),
        epsabs=epsabs,
        epsrel=epsrel,
    )[0]
    return c**3 * (rate / np.expm1(-2 * np.pi * rate)) ** 2 * I


def _local_clustering_integral_xy_symmetry(
    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
):
    rate = rtrafos.beta_to_rate(beta)

    I = dblquad(
        _local_clustering_integrand_noconstants,
        0,
        2 * np.pi,
        0,
        lambda x: x,
        args=(theta_i, r, rate),
        epsabs=epsabs,
        epsrel=epsrel,
    )[0]
    return 2 * c**3 * (rate / np.expm1(-2 * np.pi * rate)) ** 2 * I


#############################################################################
# ------------------------ TRANSFORMED INTEGRAL -----------------------------
#############################################################################


def _local_clustering_integrand_transformed(s, a, theta_i, r, rate, s_jk_calc):

    d_ij = rtrafos.angular_distance(theta_i, (s - a / 2))

    d_ik = rtrafos.angular_distance(theta_i, (s + a / 2))

    s_ij = max(1 - d_ij / (2 * np.pi * r), 0)
    s_ik = max(1 - d_ik / (2 * np.pi * r), 0)

    s_jk = s_jk_calc(a, r)

    e_j = np.exp(-rate * (s - a / 2))
    e_k = np.exp(-rate * (s + a / 2))

    return s_ij * s_ik * s_jk * e_j * e_k


def _s_jk_small(a, r):
    return 1 - a / (2 * r * np.pi)


def _s_jk_big(a, r):
    return 1 - (2 * np.pi - a) / (2 * r * np.pi)


def _local_clustering_integral_transformed(
    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
):
    rate = rtrafos.beta_to_rate(beta)

    I_a_small = dblquad(
        _local_clustering_integrand_transformed,
        0,
        2 * np.pi * r,
        lambda x: x / 2,
        lambda x: 2 * np.pi - x / 2,
        args=(theta_i, r, rate, _s_jk_small),
        epsabs=epsabs,
        epsrel=epsrel,
    )[0]

    I_a_big = dblquad(
        _local_clustering_integrand_transformed,
        2 * np.pi * (1 - r),
        2 * np.pi,
        lambda x: x / 2,
        lambda x: 2 * np.pi - x / 2,
        args=(theta_i, r, rate, _s_jk_big),
        epsabs=epsabs,
        epsrel=epsrel,
    )[0]
    I = I_a_small + I_a_big
    return 2 * c**3 * (rate / np.expm1(-2 * np.pi * rate)) ** 2 * I


#############################################################################
# ----------------------- DOUBLY TRANSFORMED INTEGRAL -----------------------
#############################################################################


def _local_clustering_integrand_dbltransformed(y, x, theta_i, r, rate, s_jk_calc):
    d_ij = rtrafos.angular_distance(x, 0)
    d_ik = rtrafos.angular_distance(y, 0)

    s_ij = max(1 - d_ij / (2 * np.pi * r), 0)
    s_ik = max(1 - d_ik / (2 * np.pi * r), 0)

    s_jk = s_jk_calc(x - y, r)

    e_j = np.exp(-rate * (theta_i - x))
    e_k = np.exp(-rate * (theta_i - y))

    return s_ij * s_ik * s_jk * e_j * e_k


def _local_clustering_integral_dbltransformed(
    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
):
    rate = rtrafos.beta_to_rate(beta)

    I_a_small1 = dblquad(
        _local_clustering_integrand_dbltransformed,
        theta_i - 2 * np.pi,
        theta_i - 2 * np.pi * (1 - r),
        theta_i - 2 * np.pi,
        lambda x: x,
        args=(theta_i, r, rate, _s_jk_small),
        epsabs=epsabs,
        epsrel=epsrel,
    )[0]
    I_a_small2 = dblquad(
        _local_clustering_integrand_dbltransformed,
        theta_i - 2 * np.pi * (1 - r),
        theta_i,
        lambda x: x - 2 * np.pi * r,
        lambda x: x,
        args=(theta_i, r, rate, _s_jk_small),
        epsabs=epsabs,
        epsrel=epsrel,
    )[0]

    I_a_big = dblquad(
        _local_clustering_integrand_dbltransformed,
        theta_i - 2 * np.pi * r,
        theta_i,
        theta_i - 2 * np.pi,
        lambda x: x - 2 * np.pi * (1 - r),
        args=(theta_i, r, rate, _s_jk_big),
        epsabs=epsabs,
        epsrel=epsrel,
    )[0]
    I = I_a_small1 + I_a_small2 + I_a_big
    return 2 * c**3 * (rate / np.expm1(-2 * np.pi * rate)) ** 2 * I
