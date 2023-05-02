import inspect
import numpy as np
import scipy.stats as ss
import scipy.special as sc

from scipy.stats._distn_infrastructure import rv_generic, rv_frozen
from ringity.classes.exceptions import DistributionParameterError

# =============================================================================
#  -------------------------- GLOBAL CONSTANTS ------------------------------
# =============================================================================


DISTN_NAME2LEMMA = {
    'normal' : 'norm',
    'exponential' : 'expon',
    'wrappedexponential' : 'wrappedexpon'
}

RINGITY_DISTN_NAMES_LIST = ['wrappedexpon']
SCIPY_DISTN_NAMES_LIST = list(filter
                (
                lambda d: isinstance(getattr(ss, d), rv_generic),
                dir(ss))
                )


# =============================================================================
#  ----------------------- RANDOM VARIABLE CLASSES ---------------------------
# =============================================================================

# PEP8 conventions have been ignored here to make the code mimic the style of
# scipy.

class wrappedexpon_gen(ss.rv_continuous):
    r"""A wrapped exponential continuous random variable with rate parameter
    ``rate``.
    """
    def _argcheck(self, rate):
        return rate >= 0

    def _pdf(self, x, rate):
        return rate*np.exp(-x*rate) / -sc.expm1(-2*np.pi*rate)

    def _cdf(self, x, rate):
        return -sc.expm1(-x*rate) / -sc.expm1(-2*np.pi*rate)

wrappedexpon = wrappedexpon_gen(a=0.0, b=2*np.pi, name='wrappedexpon')

def get_frozen_wrappedexpon(arg, argtype):
    if argtype == 'rate':
        rate = arg
    elif argtype == 'delay':
        rate =  np.tan(np.pi*(1-arg)/2)
    else:
        raise DistributionParameterError(f"Argtype `{argtype}` unknown!")
    return wrappedexpon(rate=rate)

# =============================================================================
#  ------------------------------ FUNCTIONS ----------------------------------
# =============================================================================

def _get_frozen_random_variable(distn_arg, **kwargs):
    if isinstance(distn_arg, rv_frozen):
        frozen_rv = distn_arg
    else:
        if isinstance(distn_arg, str):
            distn_name = DISTN_NAME2LEMMA.get(distn_arg, distn_arg)
            if distn_name in SCIPY_DISTN_NAMES_LIST:
                rv_gen = getattr(ss, distn_name)
            elif distn_name in RINGITY_DISTN_NAMES_LIST:
                rv_gen = eval(distn_name)
        elif isinstance(distn_arg, rv_generic):
            rv_gen = distn_arg
        else:
            raise ValueError(f'Type {type(distn_arg)} unknown.')
        frozen_rv = rv_gen(**kwargs)
        
    assert isinstance(frozen_rv, rv_frozen)
    return frozen_rv
        
def _get_rv(distn, **kwargs):
    
    if   isinstance(distn, str):
        if   distn == 'wrappedexpon':
            possible_args = inspect.getfullargspec(wrappedexpon._parse_args)[0]
            given_args = {key:value for (key,value) in kwargs.items() if key in possible_args}
            return wrappedexpon, given_args
        else:
            assert False, f"Distribution {distn} not known."
    
    elif isinstance(distn, rv_frozen):
        return ss.rv_continuous(name = 'frozen'), None
    else:
        assert False, f"data type of distn recognized: ({type(self.distribution)})"

def get_rate_parameter(parameter, parameter_type):
    if   parameter_type.lower() == 'rate':
        return parameter
    elif parameter_type.lower() == 'shape':
        return 1/parameter
    elif parameter_type.lower() == 'delay':
        return np.cos(PI*parameter/2) / np.sin(PI*parameter/2)
    else:
        assert False, f"Parameter type '{parameter_type}' not known! " \
                       "Please choose between 'rate', 'shape' and 'delay'."


def pdf_delay(t, parameter, parameter_type='rate'):
    """
    Probability distribution function of wrapped exponential distribution.
    The interpretation of `parameter` as a "rate/shape/delay parameter" is
    specified by the `parameter_type`.
    Support is on [0,2*pi].
    """
    rate = get_rate_parameter(parameter, parameter_type)
    support = np.where((0<t) & (t<2*PI), 1., 0.)
    values  = np.exp(-t*rate)
    normalization = rate / (1-np.exp(-2*PI*rate))
    return support * values * normalization

def cdf_delay(t, parameter, parameter_type='rate'):
    """
    Cumulative distribution function of wrapped exponential distribution.
    The interpretation of `parameter` as a "rate/shape/delay parameter" is
    specified by the `parameter_type`.
    Support is on [0,2*pi].
    """
    rate = get_rate_parameter(parameter, parameter_type)
    support = np.where(0<t, 1., 0.)
    values  = 1-np.exp(-t*rate)
    normalization = 1 / (1-np.exp(-2*PI*rate))
    return support * np.where(t>=2*PI, 1, values*normalization)


# =============================================================================
#  -------------------- DISTANCE DISTRIBUTION FUNCTIONS ----------------------
# =============================================================================

def pdf_absolute_distance(t, parameter, parameter_type='rate'):
    """
    Probability distribution function of (absolute) distance of two wrapped
    exponentialy distributed random variables with scale parameter kappa.
    Support is on [0,2pi].
    """
    rate = get_rate_parameter(parameter, parameter_type)
    support = np.where((0 < t) & (t < 2*PI), 1., 0.)
    values  = np.exp(-t*rate) - np.exp(rate*(t-4*PI))
    normalization = rate / (np.exp(-4*PI*rate)*(np.exp(2*PI*rate)-1)**2)
    return support * values * normalization

def cdf_absolute_distance(t, parameter, parameter_type='rate'):
    """
    Cumulative distribution function of (absolute) distance of two wrapped
    exponentialy distributed random variables with scale parameter kappa.
    Support is on [0,2pi].
    """
    rate = get_rate_parameter(parameter, parameter_type)
    support = np.where(0<t, 1., 0.)
    values  = 1 - np.exp(-rate*(4*PI-t)) + np.exp(-4*PI*rate)- np.exp(-rate*t)
    normalization = 1 / (np.exp(-4*PI*rate)*(np.exp(2*PI*rate)-1)**2)
    return support * np.where(t>=2*PI, 1, values*normalization)


def pdf_conditional_absolute_distance(t, theta,
                                      parameter,
                                      parameter_type='rate'):
    """
    [MISSING]
    """
    term1 = pdf_delay(theta+t,
                      parameter = parameter,
                      parameter_type = parameter_type)
    term2 = pdf_delay(theta-t,
                      parameter = parameter,
                      parameter_type = parameter_type)
    support = np.where((0 < t) & (t < 2*PI), 1., 0.)
    return support*(term1 + term2)

def cdf_conditional_absolute_distance(t, theta,
                                      parameter,
                                      parameter_type='rate'):
    """
    [MISSING] & somehow faulty....
    """
    term1 = cdf_delay(theta+t,
                      parameter = parameter,
                      parameter_type = parameter_type)
    term2 = cdf_delay(theta-t,
                      parameter = parameter,
                      parameter_type = parameter_type)
    support = np.where(0 < t, 1., 0.)
    return support * np.where(t >= 2*PI, 1, term1 - term2)


def pdf_circular_distance(t, parameter, parameter_type='rate'):
    """
    Probability distribution function of (circular) distance of two wrapped
    exponentialy distributed random variables with scale parameter kappa.
    Support is on [0,pi].
    """
    rate = get_rate_parameter(parameter, parameter_type)
    support = np.where((0 < t) & (t < PI), 1., 0.)
    values  = np.cosh((PI-t)*rate)
    normalization  = rate/np.sinh(PI*rate)
    return support * values * normalization

def cdf_circular_distance(t, parameter, parameter_type='rate'):
    """
    Cumulative distribution function of (circular) distance of two wrapped
    exponentialy distributed random variables with scale parameter kappa.
    F(t)=0 for t<0 and F(t)=1 for t>pi.
    """
    rate = get_rate_parameter(parameter, parameter_type)
    support = np.where(0<=t, 1., 0.)
    values = np.sinh(rate*(PI-t))
    normalization = 1 / np.sinh(rate*PI)
    return support * np.where(t >= PI, 1, 1-values*normalization)


def pdf_conditional_circular_distance(t, theta,
                                      parameter,
                                      parameter_type='rate'):
    """
    [MISSING]
    """
    term1 = pdf_conditional_absolute_distance(t, theta,
                                              parameter = parameter,
                                              parameter_type = parameter_type)
    term2 = pdf_conditional_absolute_distance(2*PI-t, theta,
                                              parameter = parameter,
                                              parameter_type = parameter_type)
    support = np.where((0 < t) & (t < PI), 1., 0.)
    return support*(term1 + term2)

def cdf_conditional_circular_distance(t, theta,
                                      parameter,
                                      parameter_type='rate'):
    """
    [MISSING]
    """
    term1 = cdf_conditional_absolute_distance(t, theta,
                                              parameter = parameter,
                                              parameter_type = parameter_type)
    term2 = cdf_conditional_absolute_distance(2*PI-t, theta,
                                              parameter = parameter,
                                              parameter_type = parameter_type)
    support = np.where(0 < t, 1., 0.)
    return support * np.where(t >= PI, 1, term1 + (1-term2))



# =============================================================================
#  ----- SIMILARITY AND INTERACTION PROBABILITY DISTRIBUTION FUNCTIONS -------
# =============================================================================

def pdf_similarity(t, parameter, a, parameter_type='rate'):
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
    rate = get_rate_parameter(parameter, parameter_type)
    s_min = np.clip(2-1/a,0,1)
    support = np.where((s_min<=t) & (t<=1), 1., 0.)
    values  = np.cosh(PI*rate * (1-2*a*(1-t)))
    normalization = 2*a*PI * rate / np.sinh(PI*rate)
    return support * values * normalization

def cdf_similarity(t, parameter, a, parameter_type='rate'):
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
    rate = get_rate_parameter(parameter, parameter_type)
    s_min = np.clip(2-1/a,0,1)
    support = np.where(s_min<=t, 1., 0.)
    numerator   = np.sinh(rate*(PI-2*a*PI*(1-t)))
    denominator = np.sinh(rate*PI)
    return support * np.where(t>=1, 1., numerator / denominator)


def pdf_conditional_similarity(t, theta, a,
                               parameter,
                               parameter_type = 'rate'):
    """
    [MISSING]
    """
    s_min = np.clip(2-1/a,0,1)
    support = np.where((s_min<=t) & (t<=1), 1., 0.)
    return support * 2*PI*a * pdf_conditional_circular_distance(
                                                t = 2*PI*a * (1-t),
                                                theta = theta,
                                                parameter = parameter,
                                                parameter_type = parameter_type)
def cdf_conditional_similarity(t, theta, a,
                               parameter,
                               parameter_type = 'rate'):
    """
    [MISSING]
    """
    assert False, "Not implemented yet!"


def pdf_probability(t, parameter, a, rho, parameter_type='rate'):
    """
    (Continuous part of the) probability density function of [MISSING]
    """
    rate = get_rate_parameter(parameter, parameter_type)
    mu_S = mean_similarity(rate,a)
    if rho <= mu_S:
        k = rho/mu_S
        s_min = np.clip(2-1/a,0,1)
        support = np.where((k*s_min<=t) & (t<=k), 1., 0.)
        values  = np.cosh(PI*rate * (1-2*a*(1-t/k)))
        normalization = 2*a*PI * rate / np.sinh(PI*rate)
        return support * values * normalization / k
    else:
        assert False, "Not implemented yet!"

def cdf_probability(t, parameter, a, rho, parameter_type='rate'):
    """
    Cumulative distribution function of [MISSING]
    """
    assert False, "Not implemented yet!"


def pdf_conditional_probability(t, theta, parameter, a, rho, parameter_type='rate'):
    """
    (Continuous part of the) probability density function of [MISSING]
    """
    rate = get_rate_parameter(parameter, parameter_type)
    mu_S = mean_similarity(rate,a)
    if rho <= mu_S:
        k = rho/mu_S
        return pdf_conditional_similarity(t/k, theta=theta, a=a,
                                          parameter = parameter,
                                          parameter_type = parameter_type)/k
    else:
        assert False, "Not implemented yet!"

def cdf_conditional_probability(t, theta, parameter, a, rho, parameter_type='rate'):
    """
    [MISSING]
    """
    assert False, "Not implemented yet!"


# =============================================================================
#  ----------------------------- MISCELLANEOUS -------------------------------
# =============================================================================

def mean_similarity(parameter, a, parameter_type='rate'):
    rate = get_rate_parameter(parameter, parameter_type)
    numerator = 2*np.sinh(PI*rate)*np.sinh((1-a)*PI*rate)*np.sinh(a*PI*rate)
    denominator = a*PI*rate*(np.cosh(2*PI*rate)-1)
    return 1 - numerator/denominator


def expected_node_degree(theta, a, rho, parameter, parameter_type = 'rate'):

    if a > 0.5:
        assert False, "Not implemented yet!"

    rate = get_rate_parameter(parameter, parameter_type)
    mu_S = mean_similarity(rate,a)
    k = rho/mu_S

    if k > 1:
        assert False, "Not implemented yet!"

    normalization = 1 / ((1-np.exp(-2*PI*rate)) * rate * (2*PI*a)**2)

    if theta <= 2*PI*a:
        const = 1 + 2*a*PI*rate - theta*rate

        term_A1 = -2 * np.exp(-rate * theta)
        term_A2 =      np.exp(-rate * (2*PI + theta - 2*a*PI))

        term_B1 =                              np.exp(-rate * (2*a*PI + theta))
        term_B2 = (theta*rate-2*a*PI*rate-1) * np.exp(-rate * 2*PI)

    else:
        term_A1 = -2 * np.exp(-rate * theta)
        term_A2 =      np.exp(-rate * (theta-2*a*PI))

        if theta + 2*a*PI <= 2*PI:
            const = 0

            term_B1 = np.exp(-rate * (2*a*PI + theta))
            term_B2 = 0

        else:
            const = theta*rate - 1 - 2*(1-a)*PI*rate

            term_B1 =                                      np.exp(-rate * (2*(a-1)*PI + theta))
            term_B2 = (1 - 2*(a-1)*PI*rate - theta*rate) * np.exp(-rate * 2*PI)

    integral = normalization * (term_A1 + term_A2 +term_B1 + term_B2 + const)

    return k*integral*2*PI*a

def d_expected_node_degree(theta, a, rho, parameter, parameter_type = 'rate'):

    if a > 0.5:
        assert False, "Not implemented yet!"

    rate = get_rate_parameter(parameter, parameter_type)
    mu_S = mean_similarity(rate,a)
    k = rho/mu_S

    if k > 1:
        assert False, "Not implemented yet!"

    normalization = 1 / ((1-np.exp(-2*PI*rate)) * rate * (2*PI*a)**2)

    if theta <= 2*PI*a:
        const = - rate

        term_A1 = 2*rate * np.exp(-rate * theta)
        term_A2 =  -rate * np.exp(-rate * (2*PI + theta - 2*a*PI))

        term_B1 =  -rate * np.exp(-rate * (2*a*PI + theta))
        term_B2 =   rate * np.exp(-rate * 2*PI)

    else:
        term_A1 = 2*rate * np.exp(-rate * theta)
        term_A2 =  -rate * np.exp(-rate * (theta-2*a*PI))

        if theta + 2*a*PI <= 2*PI:
            const = 0

            term_B1 = -rate * np.exp(-rate * (2*a*PI + theta))
            term_B2 = 0

        else:
            const = rate

            term_B1 = -rate * np.exp(-rate * (2*(a-1)*PI + theta))
            term_B2 = -rate * np.exp(-rate * 2*PI)

    integral = normalization * (term_A1 + term_A2 +term_B1 + term_B2 + const)

    return k*integral*2*PI*a

def get_max_expectancy(a, parameter, parameter_type):
    rate = get_rate_parameter(parameter=parameter, parameter_type=parameter_type)
    term = (2 - np.exp(-rate * (2*PI - 2*a*PI)) - np.exp(-rate * 2*a*PI)) / (1 - np.exp(-rate*2*PI))
    return np.log(term)/rate

def get_min_expectancy(a, parameter, parameter_type):
    rate = get_rate_parameter(parameter=parameter, parameter_type=parameter_type)
    term = (np.exp(rate * 2*a*PI) + np.exp(-rate * (2*a*PI-2*PI)) - 2) / (1 - np.exp(-rate * 2*PI))
    return np.log(term)/rate
