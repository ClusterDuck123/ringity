from numpy import pi as PI

import scipy
import numpy as np

# =============================================================================
#  ---------------- (DIS-)SIMILARITY DISTRIBUTION FUNCTIONS ------------------
# =============================================================================
def pdf_distance(t, kappa):
    """
    Cumulative distribution function of (circular) distance of two wrapped
    exponentialy distributed random variables with scale parameter kappa.
    Support is on [0,pi].
    """
    support = np.where((0<t) & (t<PI), 1., 0.)
    return support * kappa/np.sinh(PI*kappa) * np.cosh((PI-t)*kappa)

def cdf_distance(t, kappa):
    """
    Cumulative distribution function of (circular) distance of two wrapped
    exponentialy distributed random variables with scale parameter kappa.
    F(t)=0 for t<0 and F(t)=1 for t>pi.
    """
    support = np.where(0<=t, 1., 0.)
    term = 1 - np.sinh((PI-t)*kappa) / np.sinh(PI*kappa)
    return support * np.where(t>=PI,1,term)

def pdf_similarity(t, kappa, a):
    """
    Probability density function of s_a ∘ d(X,Y), where X,Y are two wrapped
    exponentially distributed random variables with scale parameter 1/kappa,
    d(-,-) denotes the distance on the circle and s_a(-) is the area of the
    (normalized) overlap of two boxes of length 2*pi*a on the circle for a
    given distance (measured from center to center). Normalization is taken
    to be the area of one box, 2*pi*a, to ensure the support being on [0,1].

    Function not implemented yet for a>0.5!
    """
    assert 0 <= a <= 0.5, f'Function not implemented yet for a>0.5! a={a}'

    support = np.where((0<=t) & (t<=1), 1., 0.)
    return support * 2*a*PI*pdf_distance(2*a*PI*(1-t), kappa)

def cdf_similarity(t, kappa, a):
    """
    Cumulative distribution function of s_a ∘ d(X,Y), where X,Y are two wrapped
    exponentially distributed random variables with scale parameter 1/kappa,
    d(-,-) denotes the distance on the circle and s_a(-) is the area of the
    (normalized) overlap of two boxes of length 2*pi*a on the circle for a
    given distance (measured from center to center). Normalization is taken
    to be the area of one box, 2*pi*a, to ensure the support being on [0,1].

    Function not implemented yet for a>0.5!
    """
    assert 0 <= a <= 0.5, f'Function not implemented yet for a>0.5! a={a}'

    support = np.where(0<=t, 1., 0.)
    term = 1 - cdf_distance(2*a*PI*(1-t), kappa=kappa)
    return support * np.where(t>=1, 1., term)

def mean_similarity(kappa, a):
    numer = 2*np.sinh(PI*kappa)*np.sinh((1-a)*PI*kappa)*np.sinh(a*PI*kappa)
    denom = a*PI*kappa*(np.cosh(2*PI*kappa)-1)
    return 1 - numer/denom
