from numpy import pi as PI

import scipy
import numpy as np

def get_rate_parameter(theta, parameter_type):
    if   parameter_type.lower() == 'rate':
        return theta
    elif parameter_type.lower() == 'shape':
        return 1/theta
    elif parameter_type.lower() == 'delay':
        return np.cos(PI*theta/2) / np.sin(PI*theta/2)
    else:
        assert False, f"Parameter type '{parameter_type}' not known! " \
                       "Please choose between 'rate', 'shape' and 'delay'."

def pdf_delay(t, theta, parameter_type='rate'):
    """
    Probability distribution function of wrapped exponential distribution.
    The interpretation of `theta` as a "rate/shape/delay parameter" is
    specified by the `parameter_type`.
    Support is on [0,2*pi].
    """
    rate = get_rate_parameter(theta, parameter_type)
    support = np.where((0<t) & (t<2*np.pi), 1., 0.)
    values  = np.exp(-t*rate)
    normalization = rate / (1-np.exp(-2*np.pi*rate))
    return support * values * normalization


def pdf_absolute_distance(t, theta, parameter_type='rate'):
    """
    Probability distribution function of (absolute) distance of two wrapped
    exponentialy distributed random variables with scale parameter kappa.
    Support is on [0,2pi].
    """
    rate = get_rate_parameter(theta, parameter_type)
    support = np.where((0<t) & (t<2*PI), 1., 0.)
    values  = np.exp(-t*rate) - np.exp(rate*(t-4*PI))
    normalization = rate / (np.exp(-4*PI*rate)*(np.exp(2*PI*rate)-1)**2)
    return support * values * normalization


# =============================================================================
#  ---------------- (DIS-)SIMILARITY DISTRIBUTION FUNCTIONS ------------------
# =============================================================================
def pdf_circular_distance(t, theta, parameter_type='rate'):
    """
    Probability distribution function of (circular) distance of two wrapped
    exponentialy distributed random variables with scale parameter kappa.
    Support is on [0,pi].
    """
    rate = get_rate_parameter(theta, parameter_type)
    support = np.where((0<t) & (t<PI), 1., 0.)
    values  = np.cosh((PI-t)*rate)
    normalization  = rate/np.sinh(PI*rate)
    return support * values * normalization

def cdf_circular_distance(t, theta, parameter_type='rate'):
    """
    Cumulative distribution function of (circular) distance of two wrapped
    exponentialy distributed random variables with scale parameter kappa.
    F(t)=0 for t<0 and F(t)=1 for t>pi.
    """
    rate = get_rate_parameter(theta, parameter_type)
    support = np.where(0<=t, 1., 0.)
    values = np.sinh(rate*(PI-t))
    normalization = 1 / np.sinh(rate*PI)
    return support * np.where(t >= PI, 1, 1-values*normalization)

def pdf_similarity(t, theta, a, parameter_type='rate'):
    """
    (Continuous part of the) probability density function of s_a ∘ d(X,Y),
    where X,Y are two wrapped exponentially distributed random variables
    with scale parameter kappa, d(-,-) denotes the distance on the circle
    and s_a(-) is the area of the (normalized) overlap of two boxes of length
    2*pi*a on the circle for a given distance (measured from edge to edge,
    or equivalently from center to center).
    Normalization is taken to be the area of one box, 2*pi*a. Hence, the
    support is on [s_min,pi], where s_min=|2-1/a|^+.
    """
    rate = get_rate_parameter(theta, parameter_type)
    s_min = np.clip(2-1/a,0,1)
    support = np.where((s_min<=t) & (t<=1), 1., 0.)
    values  = np.cosh(PI*rate * (1-2*a*(1-t)))
    normalization = 2*a*PI * rate / np.sinh(PI*rate)
    return support * values * normalization

def cdf_similarity(t, theta, a, parameter_type='rate'):
    """
    Cumulative distribution function of s_a ∘ d(X,Y), where X,Y are two wrapped
    exponentially distributed random variables with scale parameter kappa,
    d(-,-) denotes the distance on the circle and s_a(-) is the area of the
    (normalized) overlap of two boxes of length 2*pi*a on the circle for a
    given distance (measured from edge to edge, or equivalently from center
    to center).
    Normalization is taken to be the area of one box, 2*pi*a. Hence, the
    support is on [s_min,pi], where s_min=|2-1/a|^+.
    """
    rate = get_rate_parameter(theta, parameter_type)
    s_min = np.clip(2-1/a,0,1)
    support = np.where(s_min<=t, 1., 0.)
    numerator   = np.sinh(rate*(PI-2*a*PI*(1-t)))
    denominator = np.sinh(rate*PI)
    return support * np.where(t>=1, 1., numerator / denominator)

def pdf_probability(t, theta, a, rho, parameter_type='rate'):
    """
    (Continuous part of the) probability density function of [MISSING]
    """
    rate = get_rate_parameter(theta, parameter_type)
    mu_S = mean_similarity(rate,a)
    if rho <= mu_S:
        k = rho/mu_S
        s_min = np.clip(2-1/a,0,1)
        support = np.where((k*s_min<=t) & (t<=k), 1., 0.)
        values  = np.cosh(PI*rate * (1-2*a*(1-t/k)))
        normalization = 2*a*PI * rate / np.sinh(PI*rate)
        return support * values * normalization / k

def cdf_probability(t, theta, a, rho, parameter_type='rate'):
    """
    Cumulative distribution function of [MISSING]
    """
    rate = get_rate_parameter(theta, parameter_type)
    mu_S = mean_similarity(rate,a)

    if rho <= mu_S:
        k = rho/mu_S
        s_min = np.clip(2-1/a,0,1)
        support = np.where(s_min<=t, 1., 0.)
        numerator   = np.sinh(rate*(PI-2*a*PI*(1-t/k)))
        denominator = np.sinh(rate*PI)
        return support * np.where(t>=1, 1., numerator / denominator)

def mean_similarity(theta, a, parameter_type='rate'):
    rate = get_rate_parameter(theta, parameter_type)
    numerator = 2*np.sinh(PI*rate)*np.sinh((1-a)*PI*rate)*np.sinh(a*PI*rate)
    denominator = a*PI*rate*(np.cosh(2*PI*rate)-1)
    return 1 - numerator/denominator
