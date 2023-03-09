import scipy
import numpy as np

def slope(rho, rate, a):
    mu_S = mean_similarity(rate,a)
    if rho <= mu_S:
        return rho/mu_S
    else:
        const = 1/np.sinh(np.pi*rate)
        def integral(k):
            term1 = np.sinh((1 + 2*a*(1/k-1))*np.pi*rate)
            term2 = (k*np.sinh((a*np.pi*rate)/k)*np.sinh(((a+k-2*a*k)*np.pi*rate)/k))/(a*np.pi*rate)
            return term1-term2
        return scipy.optimize.newton(
            func = lambda k: const*integral(k) + (1-cdf_similarity(1/k, rate, a)) - rho,
            x0 = rho/mu_S)