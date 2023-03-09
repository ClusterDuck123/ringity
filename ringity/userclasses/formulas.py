"""
THIS MODULE IS STILL EXPERIMENTAL!
"""

import numpy as np

class Formula():
    pass

class Parameter(Formula):
    pass

class AllParameters(Formula):
    def __init__(self, response, coupling, rate, density, N) -> None:
        self._response = response
        self._coupling = coupling
        self._rate = rate
        self._density = density
        self._N

    @property
    def N(self):
        return self._N
    @property
    def response(self):
        return self._response
    @property
    def coupling(self):
        return self._coupling
    @property
    def rate(self):
        return self._rate
    @property
    def density(self):
        return self._density
    
    def max_density(self):
        """Calculate maximal allowed density (equivalently expected mean similarity).

        This function assumes a (wrapped) exponential function and a cosine similarity
        of box functions.
        """

        if np.isclose(self.rate, 0):
            return self.response
            
        if np.isclose(self.response, 0):
            return 0

        # Python can't handle this case properly;
        # maybe the analytical expression can be simplified...
        if self.rate > 200:
            return 1.

        plamb = np.pi * self.rate

        # Alternatively:
        # numerator = 2*(np.sinh(plamb*(1-a)) * np.sinh(plamb*a))
        numerator = (np.cosh(plamb) - np.cosh(plamb*(1 - self.response*2)))
        denominator = (self.response*2*plamb * np.sinh(plamb))

        return 1 - numerator / denominator
    
    def governing_equation(self):
        rho_max = self.max_density()
        return self.coupling*rho_max  - self.density
        
class Distribution(Parameter):
    def rate_to_beta(self):
        beta = 1 - 2/np.pi * np.arctan(self.rate)
        return beta
    def beta_to_rate(self, beta):
        if np.isclose(beta,0):
            rate = np.inf
        else:
            rate = np.tan(np.pi * (1-beta) / 2)
        return rate
    

class Density(Parameter):
    def __init__(self, density, coupling, rate, N=None) -> None:
        super().__init__(density, coupling, rate, N)
    def coupling_to_density(coupling, response, rate):
        density = coupling * max_density(response = response, rate = rate)
        return density

class Coupling(Parameter):
    pass


