from numpy import pi as PI

import scipy
import numpy as np

# =============================================================================
#  ---------------------------  VON MISES FUNCTIONS -------------------------
# =============================================================================

#  ------------------------------- preparation -------------------------------
def bessel_I0(x):
    """Modified Bessel function of the first kind of order 0"""
    return scipy.special.iv(0,x)

#  --------------------------- von Mises functions ---------------------------
def d_mi(t, kappa):
    """Derivative of 'Little Me function'. Main function to express the pdf of
    the difference of two von Mises distributed random variables."""
    return bessel_I0(2*kappa*np.cos(t/2))

def mi(x, kappa, limit=100):
    """
    Little Mi (ⲙ) function. Main function to express the cdf of the difference
    of two von Mises distributed random variables. The keyword 'limit' is the
    maximal number of iteration for the integral.
    """

# Error of Taylor approximation has been tested to be below (Python) floating
# point precision for kappa < 1. It's ugly af but much faster than the integral
# implementation below.
    if kappa <= 1:
        x = np.array(x)
        integrals = np.zeros((6,*x.shape))
        integrals[0] = x
        integrals[1] = (x + np.sin(x))/2
        integrals[2] = (   3*x + np.sin(x)*(np.cos(x) + 4))/8
        integrals[3] = (  30*x +   45*np.sin(x) +   9*np.sin(2*x) +     np.sin(3*x))/96
        integrals[4] = ( 420*x +  672*np.sin(x) + 168*np.sin(2*x) +  32*np.sin(3*x) +  3*np.sin(4*x))/1536
        integrals[5] = (1260*x + 2100*np.sin(x) + 600*np.sin(2*x) + 150*np.sin(3*x) + 25*np.sin(4*x) + 2*np.sin(5*x))/5120

        return sum((kappa**m / np.math.factorial(m))**2 * integrals[m] for m in range(6))
    else:
        def integral(upper_limit): # define integral for vectorization.
            return scipy.integrate.quad(lambda t : d_mi(t, kappa),
                                        a = 0,
                                        b = upper_limit,
                                        limit = limit)[0]
        return np.vectorize(integral)(x)

def Mi(x, kappa, limit=100):
    """
    Big Mi (Ⲙ) function. Main function to express something.
    [PROBABLY DEPRICATED]
    """
    return scipy.integrate.quad(lambda t : mi(t, kappa=kappa), 0, x,
                                limit=limit)[0]


# =============================================================================
#  ---------------- (DIS-)SIMILARITY DISTRIBUTION FUNCTIONS ------------------
# =============================================================================
def pdf_distance(t ,kappa):
    """
    Cumulative distribution function of (circular) distance of two von Mises
    distributed random variables with concentration parameter kappa.
    Support is on [0,pi].
    """
    support = np.where((0<t) & (t<PI), 1., 0.)
    return support * d_mi(t,kappa) / (PI*bessel_I0(kappa)**2)

def cdf_distance(t, kappa):
    """
    Cumulative distribution function of (circular) distance of two von Mises
    distributed random variables with concentration parameter kappa.
    F(t)=0 for t<0 and F(t)=1 for t>pi.
    """
    support = np.where(0<=t, 1., 0.)
    return support * np.where(t>=PI, 1.,
                              mi(t,kappa)/(PI*bessel_I0(kappa)**2))
def pdf_similarity(t, a, kappa):
    """
    Probability density function of s_a ∘ d(X,Y), where X,Y are two von
    Mises distributed random variables with concentration parameter kappa,
    d(-,-) denotes the distance on the circle and s_a(-) is the area of the
    (normalized) overlap of two boxes of length 2*pi*a on the circle for a given
    distance (measured from center to center). Normalization is taken to be the
    area of one box, 2*pi*a, to ensure the support being on [0,1].

    Function not implemented yet for a>0.5!
    """

    assert 0 <= a <= 0.5, f'Function not implemented yet for a>0.5! a={a}'

    support = np.where((0<=t) & (t<=1), 1., 0.)
    return support * 2*a*PI*pdf_distance(2*a*PI*(1-t), kappa=kappa)

def cdf_similarity(t, a, kappa):
    """
    Cumulative distribution function of s_a ∘ d(X,Y), where X,Y are two von
    Mises distributed random variables with concentration parameter kappa,
    d(-,-) denotes the distance on the circle and s_a(-) is the area of the
    (normalized) overlap of two boxes of length 2*pi*a on the circle for a given
    distance (measured from center to center). Normalization is taken to be the
    area of one box, 2*pi*a, to ensure the support being on [0,1].

    Function not implemented yet for a>0.5!
    """

    assert 0 <= a <= 0.5, f'Function not implemented yet for a>0.5! a={a}'

    support = np.where(0<=t, 1., 0.)
    return support * np.where(t>=1, 1.,
                              1 - cdf_distance(2*a*PI*(1-t), kappa=kappa))


# =============================================================================
#  ------------------- CONNECTION PROBABILITY FUNCTIONS ----------------------
# =============================================================================
"""Nomeclature is inspired by [1]"""

#  ------------------------------- preparation -------------------------------
def mueta_to_alphabeta(mu, eta):
    """
    Returns parameters alpha and beta of a Beta distribution with mu = mu and
    var = eta * mu * (1-mu).
    """
    nu = (1-eta)/eta
    alpha =   mu  *nu
    beta  = (1-mu)*nu
    return alpha, beta

def get_mu(rho, eta, a, kappa, maxiter=100):
    """
    Calculates the (unique) value mu of a Beta distributed random variable B
    with mean value mu = mu, and variance var = eta * mu * (1-mu), such that
    E[F_B ∘ s_a ∘ d(X,Y)] = rho, E denotes the expected value, F_B the cdf of B
    and where X, Y, d and s_a are definded in pdf_similarity.
    Function not implemented yet for a>0.5!
    """

    assert 0 <= a <= 0.5, f'Function not implemented yet for a>0.5! a={a}'

# For kappa = 0, the von Mises distribution becomes a uniform distribution on
# the circle. Hence d(X,Y) is a uniform distribution on [0,pi] and s_a ∘ d(X,Y)
# becomes a uniform distribution with a discrete part at 0. This simplifies
# calculations greatly.
    if kappa == 0:
        return 1-rho/(2*a)

# For eta = 0, the Beta distribution becomes a Delta distribution centered at
# mu. The simpified equation is now F_{similarity}(mu) = 1-rho.
# This is equivalent to F_{distance}(2*a*PI*(1-mu)) = rho.
    elif eta == 0:
        return scipy.optimize.newton(
            func = lambda mu: cdf_distance(2*a*PI*(1-mu), kappa=kappa) - rho,
            x0 = 1-rho/(2*a),
            fprime = lambda mu: -2*a*PI*pdf_distance(2*a*PI*(1-mu), kappa=kappa),
            maxiter = maxiter)

# For eta = 1, the Beta distribution becomes a Bernoulli distribution with
#expected value being mu. This simplifis some calculations. [INTUITION MISSING!]
    elif eta == 1:
        F0 = cdf_similarity(0, a=a, kappa=kappa)
        return (1-rho-F0)/(1-F0)

# General case, a.k.a. I didn't bother anymore to simplify the integral.
# [CLEAN-UP, fprime IS MISSING FOR SPEED-UP]
    else:
        def integral(mu, eta, a, kappa):
            alpha, beta = mueta_to_alphabeta(mu=mu, eta=eta)
            return 2*PI*a*scipy.integrate.quad(
                func = lambda t: scipy.stats.beta.cdf(t, a=alpha, b=beta) * \
                                        pdf_distance(2*PI*a*(1-t), kappa=kappa),
                a = 0,
                b = 1)[0]
        return scipy.optimize.newton(
            func = lambda mu: integral(mu=mu, eta=eta, a=a, kappa=kappa) - rho,
            x0 = 1-rho/(2*a),
            maxiter=100)
    return mu

#  -------------- connection probability probability functions ---------------
def pdf_connection_probability(t, rho, eta, a, kappa):
    """
    Probability density function of H_{rho, eta, a} ∘ d(X,Y;a), where X,Y are
    two von Mises distributed random variables H is the connection (probability)
    function H = F_B ∘ s_a, where s_a is s_a is definded in pdf_similarity and
    F_B in get_mu respectively. Support is on [0,1].

    Function not implemented yet for a>0.5!
    """

    assert 0 <= a <= 0.5, f'Function not implemented yet for a>0.5! a={a}'

# For eta = 0, the Beta distributed random variable B becomes a deterministic
# (Delta distributed) and subsequently H ∘ d(X,Y) is binomially distributed
# (why?). As a purely discrete distribution the 'continues' part of the pdf is
# constntly zero
    if eta == 0:
        return np.zeros_like(t)
    else:
        mu = get_mu(rho=rho, eta=eta, a=a, kappa=kappa)
        alpha, beta = mueta_to_alphabeta(mu=mu, eta=eta)

        ppf = np.nan_to_num(scipy.stats.beta.ppf(t, a=alpha, b=beta))

        numer = pdf_similarity(ppf, kappa=kappa, a=a)
        denom = scipy.stats.beta.pdf(ppf, a=alpha, b=beta)

        support = np.where((0<t) & (t<1), 1., 0.)
        return support * np.true_divide(numer, denom,
                                        out   = np.zeros_like(numer),
                                        where = denom!=0)


def cdf_connection_probability(t, rho, eta, a, kappa):
    """
    Cumulative distribution function of H_{rho, eta, a} ∘ d(X,Y;a), where X,Y
    are two von Mises distributed random variables H is the connection
    (probability) function H = F_B ∘ s_a, where s_a is s_a is definded in
    pdf_similarity and F_B in get_mu respectively.
    F(t)=0 for t<0 and F(t)=1 for t>pi.

    Function not implemented yet for a>0.5!
    """

    assert 0 <= a <= 0.5, f'Function not implemented yet for a>0.5! a={a}'

# For eta = 0, the Beta distributed random variable B becomes a deterministic
# (Delta distributed) and subsequently H ∘ d(X,Y) is binomially distributed
# (why?).
    if eta == 0:
        ppf = np.where(t < 1, 1-rho, 1.)
    else:
        mu = get_mu(rho=rho, eta=eta, a=a, kappa=kappa)
        alpha, beta = mueta_to_alphabeta(mu=mu, eta=eta)

        ppf = np.nan_to_num(scipy.stats.beta.ppf(t, a=alpha, b=beta), nan=1.)

    support = np.where(0<=t, 1., 0.)
    return support * cdf_similarity(ppf, a=a, kappa=kappa)


# =============================================================================
#  ------------------------------ REFERENCES ---------------------------------
# =============================================================================
"""
[1] C. P. Dettmann, O. Georgiou;
    Random geometric graphs with general connection functions
"""
