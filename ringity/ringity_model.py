from numpy import pi as PI

import scipy
import numpy as np

# =============================================================================
#  ------------------------------  ME-FUNCTIONS ------------------------------
# =============================================================================

def bessel(x):
    """Modified Bessel function of the first kind of order 0"""
    return scipy.special.iv(0,x)

def dvM(t, kappa):
    """Tiny Me."""
    return bessel(2*kappa*np.cos(t/2))

def vM(x, kappa, limit=100):
    """Little Me."""
    if kappa <= 1:
        x = np.array(x)
        integrals = np.zeros((6,*x.shape))
        integrals[0] = x
        integrals[1] = (x + np.sin(x))/2
        integrals[2] = (3*x + np.sin(x)*(np.cos(x) + 4))/8
        integrals[3] = (  30*x +   45*np.sin(x) +   9*np.sin(2*x) +     np.sin(3*x))/96
        integrals[4] = ( 420*x +  672*np.sin(x) + 168*np.sin(2*x) +  32*np.sin(3*x) +  3*np.sin(4*x))/1536
        integrals[5] = (1260*x + 2100*np.sin(x) + 600*np.sin(2*x) + 150*np.sin(3*x) + 25*np.sin(4*x) + 2*np.sin(5*x))/5120

        return sum((kappa**m / np.math.factorial(m))**2 * integrals[m] for m in range(6))
    else:
        def integral(upper_limit):
            return scipy.integrate.quad(lambda t : bessel(2*kappa*np.cos(t/2)), 0, upper_limit,
                                limit=limit)[0]
        return np.vectorize(integral)(x)

def SvM(x, kappa, limit=100):
    """Big Me."""
    return scipy.integrate.quad(lambda t : vM(t, kappa=kappa), 0, x,
                                limit=limit)[0]


# =============================================================================
#  ---------------- (DIS-)SIMILARITY DISTRIBUTION FUNCTIONS ------------------
# =============================================================================
def f_delta(t ,kappa):
    """pdf of pairwise distances. Support is in [0,pi]."""
    return np.where(t<=0, 0.,
                    np.where(t>=PI, 0,
                             dvM(t,kappa=kappa)/(PI*bessel(kappa)**2)))
def F_delta(t, kappa):
    """cdf of pairwise distances. Function is 0 for t<0 and 1 for t>pi."""
    return np.where(t<=0, 0.,
                    np.where(t>=PI, 1,
                             vM(t, kappa=kappa)/(PI*bessel(kappa)**2)))
def f_s(t, kappa, a):
    """pdf of pairwise similarity. Support is in [0,1]."""
    return np.where(t<=0, 0.,
                    2*a*PI*f_delta(2*a*PI*(1-t), kappa=kappa))

def F_s(t, kappa, a):
    """cdf of pairwise distances. Function is 0 for t<0 and 1 for t>1."""
    return np.where(t<=0, 0.,
                    1 - F_delta(2*a*PI*(1-t), kappa=kappa))

# --------------------------- this is a subsection ----------------------------

# =============================================================================
#  ------------------ MAIN (IF THIS IS A STANDALONE SCRIPT) ------------------
# =============================================================================

def main():
    pass

# =============================================================================
#  ----------------------------------- INIT -----------------------------------
# =============================================================================
