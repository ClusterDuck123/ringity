import numpy as np
import scipy.stats as ss
import scipy.special as sc

from .transformations import _get_rate
from scipy.stats._distn_infrastructure import _ShapeInfo


def _get_support(condition):
    support = np.where(condition, 1.0, 0.0)
    return support


def _compute_rate_and_support(condition, beta=None, rate=None):
    rate = _get_rate(beta, rate)
    support = _get_support(condition)
    return rate, support


#  ----------------------------- DELAY DISTRIBUTION ---------------------------


def pdf_delay(x, beta=None, rate=None):
    """
    Probability distribution function of wrapped exponential distribution.
    Support is on [0,2*pi].
    """
    rate, support = _compute_rate_and_support(
        condition=(0 < x) & (x < 2 * np.pi),
        beta=beta,
        rate=rate,
    )
    if rate == 0:
        expression = 1 / (2 * np.pi)
    elif rate == np.inf:
        raise ValueError("PDF does not exist for `rate == np.inf`!")
    else:
        expression = rate * np.exp(-x * rate) / (-sc.expm1(-2 * np.pi * rate))
    return expression * support


def cdf_delay(x, beta=None, rate=None):
    """
    Cumulative distribution function of wrapped exponential distribution.
    Support is on [0,2*pi].
    """
    rate, support = _compute_rate_and_support(
        condition=(0 <= x),
        beta=beta,
        rate=rate,
    )
    if rate == 0:
        expression = x / (2 * np.pi)
    elif rate == np.inf:
        expression = 1
    else:
        expression = sc.expm1(-x * rate) / sc.expm1(-2 * np.pi * rate)
    return expression * support


def ppf_delay(q, beta=None, rate=None):
    rate, support = _compute_rate_and_support(
        condition=(0 <= q),
        beta=beta,
        rate=rate,
    )
    if rate == 0:
        expression = q * (2 * np.pi)
    elif rate == np.inf:
        raise ValueError("PPF does not exist for `rate == np.inf`!")
    else:
        expression = -sc.log1p(q * sc.expm1(-2 * np.pi * rate)) / rate
    return expression * support


class delaydist_gen(ss.rv_continuous):
    r"""A wrapped exponential continuous random variable."""

    def _argcheck(self, beta):
        return (0 <= beta) & (beta <= 1)

    def _shape_info(self):
        return [_ShapeInfo("beta", False, (0, 1), (True, True))]

    def _get_support(self, beta):
        return self.a, self.b

    def _pdf(self, x, beta):
        return np.vectorize(pdf_delay)(x, beta=beta)

    def _cdf(self, x, beta):
        return np.vectorize(cdf_delay)(x, beta=beta)

    def _ppf(self, q, beta):
        return np.vectorize(ppf_delay)(q, beta=beta)


delaydist = delaydist_gen(a=0.0, b=2 * np.pi, name="delaydist")

#  ----------------------------- DISTANCE DISTRIBUTION ---------------------------


def pdf_distance(x, beta=None, rate=None):
    """
    Probability distribution function of (circular) distance of two wrapped
    exponentialy distributed random variables with scale parameter kappa.
    Support is on [0,pi].
    """
    rate, support = _compute_rate_and_support(
        condition=(0 < x) & (x < np.pi),
        beta=beta,
        rate=rate,
    )
    if rate == 0:
        expression = 1 / np.pi
    elif rate == np.inf:
        raise ValueError("PDF does not exist for `rate == np.inf`!")
    else:
        values = np.cosh((np.pi - x) * rate)
        normalization = rate / np.sinh(np.pi * rate)
        expression = values * normalization
    return expression * support


def cdf_distance(x, beta=None, rate=None):
    """
    Cumulative distribution function of (circular) distance of two wrapped
    exponentialy distributed random variables.
    F(t)=0 for t<0 and F(t)=1 for t>pi.
    """
    rate, support = _compute_rate_and_support(
        condition=(0 <= x),
        beta=beta,
        rate=rate,
    )
    if rate == 0:
        expression = x / np.pi
    elif rate == np.inf:
        expression = 1
    else:
        values = np.sinh(rate * (np.pi - x))
        normalization = 1 / np.sinh(rate * np.pi)
        expression = 1 - values * normalization
    return expression * support


class distancedist_gen(ss.rv_continuous):
    r"""A wrapped exponential continuous random variable."""

    def _shape_info(self):
        return [_ShapeInfo("beta", False, (0, 1), (True, True))]

    def _get_support(self, beta):
        return self.a, self.b

    def _pdf(self, x, beta):
        return np.vectorize(pdf_distance)(x, beta=beta)

    def _cdf(self, x, beta):
        return np.vectorize(cdf_distance)(x, beta=beta)


distancedist = distancedist_gen(a=0.0, b=np.pi, name="distancedist")


def pdf_conditional_distance(x, theta, beta=None, rate=None):
    """
    Probability distribution function of conditional (circular) distance
    of two wrapped exponentialy distributed random variables with scale
    parameter kappa. Support is on [0,pi].
    """
    rate, support = _compute_rate_and_support(
        condition=(0 < x) & (x < np.pi),
        beta=beta,
        rate=rate,
    )
    theta0 = np.pi - abs(theta - np.pi)
    eps = np.sign(theta - np.pi)
    values = np.where(
        x <= theta0,
        np.exp(-eps * rate * np.pi) * np.cosh(rate * x),
        np.cosh(rate * (x - np.pi)),
    )
    normalization = rate * np.exp(eps * rate * theta0) / np.sinh(np.pi*rate)
    return support * values * normalization


def cdf_conditional_distance(x, theta, beta=None, rate=None):
    """
    Cumulative distribution function of conditional (circular) distance
    of two wrapped exponentialy distributed random variables with scale
    parameter kappa. Support of the corresponding pdf is on [0,pi].
    """
    rate, support = _compute_rate_and_support(
        condition=(0 <= x),
        beta=beta,
        rate=rate,
    )
    theta0 = np.pi - abs(theta - np.pi)
    eps = np.sign(theta - np.pi)
    values = np.where(
        x <= theta0,
        np.sinh(rate * x),
        np.sinh(rate * theta0)
        + np.exp(eps * np.pi * rate)
        * (np.sinh(rate * (np.pi - theta0)) - np.sinh(rate * (np.pi - x))),
    )
    normalization = 2 * np.exp(-rate * theta) / (-sc.expm1(-2 * np.pi * rate))
    return support * values * normalization


class conddistancedist_gen(ss.rv_continuous):
    r"""A wrapped exponential continuous random variable."""

    def _shape_info(self):
        return [
            _ShapeInfo("beta", False, (0, 1), (True, True)),
            _ShapeInfo("theta", False, (0, 2 * np.pi), (True, True)),
        ]

    def _get_support(self, beta, theta):
        return self.a, self.b

    def _pdf(self, x, beta, theta):
        return np.vectorize(pdf_conditional_distance)(x, beta=beta, theta=theta)

    def _cdf(self, x, beta, theta):
        return np.vectorize(cdf_conditional_distance)(x, beta=beta, theta=theta)


conddistancedist = conddistancedist_gen(a=0.0, b=np.pi, name="conddistancedist")


#  ----------------------------- SIMILARITY DISTRIBUTION ---------------------------


def pdf_similarity(t, r, beta=None, rate=None):
    """
    (Continuous part of the) probability density function of s_r ∘ d(X,Y),
    where X,Y are two wrapped exponentially distributed random variables
    with delay parameter beta, d(-,-) denotes the circular distance and s_r is the
    (normalized) area of the overlap of two boxes of length 2*pi*r on the circle.
    Normalization is taken to be the area of one box, 2*pi*r. Hence, the
    support is on [s_min,1], where s_min=|1 - (1-r)/r|_+.
    """
    l = 2 * np.pi * r  # Response length
    s_min = np.clip(1 - (1 - r) / r, 0, 1)

    rate, support = _compute_rate_and_support(
        condition=(s_min <= t) & (t <= 1),
        beta=beta,
        rate=rate,
    )
    values = np.cosh(rate * (np.pi - l * (1 - t)))
    normalization = l * rate / np.sinh(np.pi * rate)
    return support * values * normalization

def pdf_unnormalized_similarity(x, beta, r):
    rate, support = _compute_rate_and_support(
        condition=(0 <= x) & (x <= 2*np.pi*r),
        beta=beta,
        rate=None,
    )
    values = np.cosh(rate * (np.pi*(1 - 2 * r) + x))
    normalization = rate / np.sinh(np.pi * rate)
    return support * values * normalization


def cdf_similarity(t, r, beta=None, rate=None):
    """
    Cumulative density function of s_r ∘ d(X,Y),
    where X,Y are two wrapped exponentially distributed random variables
    with delay parameter beta, d(-,-) denotes the circular distance and s_r is the
    (normalized) area of the overlap of two boxes of length 2*pi*r on the circle.
    Normalization is taken to be the area of one box, 2*pi*r. Hence, the
    support is on [s_min,infinity], where s_min=|1 - (1-r)/r|_+.
    """
    l = 2 * np.pi * r  # Response length
    s_min = np.clip(1 - (1 - r) / r, 0, 1)  # Minimal similarity

    rate, support = _compute_rate_and_support(
        condition=(s_min <= t),
        beta=beta,
        rate=rate,
    )

    values = np.sinh(rate * (np.pi - l * (1 - t)))
    normalization = 1 / np.sinh(rate * np.pi)
    return support * values * normalization


class similaritydist_gen(ss.rv_continuous):
    r"""A wrapped exponential continuous random variable."""

    def _shape_info(self):
        return [
            _ShapeInfo("beta", False, (0, 1), (True, True)),
            _ShapeInfo("r", False, (0, 1), (True, True)),
        ]

    def _get_support(self, beta, r):
        return self.a, self.b

    def _pdf(self, x, beta, r):
        return np.vectorize(pdf_similarity)(x, beta=beta, r=r)

    def _cdf(self, x, beta, r):
        return np.vectorize(cdf_similarity)(x, beta=beta, r=r)


similaritydist = similaritydist_gen(a=0.0, b=1.0, name="similaritydist")


def pdf_conditional_similarity(x, r, theta, beta=None, rate=None):
    """
    (Continuous part of the) probability density function of s_r ∘ d(X,theta),
    where X is a wrapped exponentially distributed random variable
    with delay parameter beta, d(-,-) denotes the circular distance and s_r is the
    (normalized) area of the overlap of two boxes of length 2*pi*r on the circle.
    Normalization is taken to be the area of one box, 2*pi*r. Hence, the
    support is on [s_min,1], where s_min=|1 - (1-r)/r|_+.
    """
    l = 2 * np.pi * r  # Response length
    s_min = np.clip(1 - (1 - r) / r, 0, 1)

    rate, support = _compute_rate_and_support(
        condition=(s_min < x) & (x < 1),
        beta=beta,
        rate=rate,
    )

    theta0 = np.pi - abs(theta - np.pi)
    eps = np.sign(theta - np.pi)

    values = np.where(
        1 - theta0 / l <= x,
        np.cosh(rate * l * (1 - x)),
        np.exp(eps * rate * np.pi) * np.cosh(rate * (l * (1 - x) - np.pi)),
    )
    normalization = (
        2 * rate * l * np.exp(-rate * theta) / (-sc.expm1(-2 * np.pi * rate))
    )
    return support * values * normalization


def cdf_conditional_similarity(x, r, theta, beta=None, rate=None):
    """
    (Continuous part of the) probability density function of s_r ∘ d(X,theta),
    where X is a wrapped exponentially distributed random variable
    with delay parameter beta, d(-,-) denotes the circular distance and s_r is the
    (normalized) area of the overlap of two boxes of length 2*pi*r on the circle.
    Normalization is taken to be the area of one box, 2*pi*r. Hence, the
    support is on [s_min,1], where s_min=|1 - (1-r)/r|_+.
    """
    l = 2 * np.pi * r  # Response length
    s_min = np.clip(1 - (1 - r) / r, 0, 1)

    rate, support = _compute_rate_and_support(
        condition=(s_min < x),
        beta=beta,
        rate=rate,
    )

    theta0 = np.pi - abs(theta - np.pi)
    eps = np.sign(theta - np.pi)

    values = np.where(
        l * (1 - x) <= theta0,
        np.sinh(rate * l * (1 - x)),
        np.exp(eps * rate * (np.pi - theta0)) * np.sinh(rate * np.pi)
        - np.exp(eps * rate * np.pi) * np.sinh(rate * (np.pi - l * (1 - x))),
    )
    normalization = 2 * np.exp(-rate * theta) / (-sc.expm1(-2 * np.pi * rate))
    return support * (1 - values * normalization)


class condsimilaritydist_gen(ss.rv_continuous):
    r"""A wrapped exponential continuous random variable."""

    def _shape_info(self):
        return [
            _ShapeInfo("theta", False, (0, 2 * np.pi), (True, True)),
            _ShapeInfo("beta", False, (0, 1), (True, True)),
            _ShapeInfo("r", False, (0, 1), (True, True)),
        ]

    def _get_support(self, theta, beta, r):
        return self.a, self.b

    def _pdf(self, x, theta, beta, r):
        return np.vectorize(pdf_conditional_similarity)(x, theta=theta, r=r, beta=beta)

    def _cdf(self, x, theta, beta, r):
        return np.vectorize(cdf_conditional_similarity)(x, theta=theta, r=r, beta=beta)


condsimilaritydist = condsimilaritydist_gen(a=0.0, b=1.0, name="condsimilaritydist")


#  ----------------------------- AUXILIARY DISTRIBUTIONS ---------------------------


def pdf_conditional_absolute_distance(t, theta, beta=None, rate=None):
    """
    Probability distribution function of conditional absolute distance
    of two wrapped exponentialy distributed random variables with scale
    parameter kappa. Support is on [0,2*pi].
    """
    rate, support = _compute_rate_and_support(
        condition=(0 < t) & (t < np.pi),
        beta=beta,
        rate=rate,
    )
    theta0 = np.pi - abs(theta - np.pi)
    eps = np.sign(theta - np.pi)
    values = np.where(t <= theta0, 2 * np.cosh(rate * t), np.exp(eps * rate * t))
    normalization = rate * np.exp(-rate * theta) / (-sc.expm1(-2 * np.pi * rate))
    support = np.where((0 < t) & (t < 2 * np.pi - theta0), 1.0, 0.0)
    return support * values * normalization