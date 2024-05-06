import numpy as np
from ringity.networkmodel.transformations import rate_to_beta
from ringity.networkmodel.param_utils import (
    governing_equation,
    infer_response_parameter,
    infer_coupling_parameter,
    infer_rate_parameter,
    infer_density_parameter,
)

"""This module describes the ParameterBuilder class used to deal
with various parameter combinations in the NetworkBuilder class.
"""


class ModelParameterBuilder:
    """The ParameterBuilder class is used to deal with various parameter
    combinations for the NetworkBuilder class.
    """

    def __init__(
        self,
        size=None,
        response=None,
        rate=None,
        coupling=None,
        density=None,
        distribution=None,
    ):
        self._size = size
        self._response = response
        self._rate = rate
        self._coupling = coupling
        self._density = density
        self._distribution = distribution

    # ------------------------------------------------------------------
    # --------------------------- PROPERTIES ---------------------------
    # ------------------------------------------------------------------
    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        if value is not None:
            if not (0 <= value):
                raise AttributeError(f"Size value {value} is invalid!")
        self._size = value

    @property
    def rate(self):
        return self._rate

    @property
    def delay(self):
        return rate_to_beta(self._rate)

    @rate.setter
    def rate(self, value):
        if value is not None:
            if not (0 <= value):
                raise AttributeError(f"Rate value {value} is invalid!")
        self._rate = value

    @property
    def response(self):
        return self._response

    @response.setter
    def response(self, value):
        if value not in {None, np.nan}:
            if not (0 <= value <= 1):
                raise AttributeError(f"Response value {value} is invalid!")
        self._response = value

    @property
    def coupling(self):
        return self._coupling

    @coupling.setter
    def coupling(self, value):
        if value is not None:
            if not (0 <= value <= 1):
                raise AttributeError(f"Density value {value} is invalid!")
        self._coupling = value

    @property
    def density(self):
        return self._density

    @density.setter
    def density(self, value):
        if value is not None:
            if not (0 <= value <= 1):
                raise AttributeError(f"Density value {value} is invalid!")
        self._density = value

    # ------------------------------------------------------------------
    # ---------------------------- METHODS -----------------------------
    # ------------------------------------------------------------------

    def infer_missing_parameters(self, verbose=False):
        """Completes the set of network paramters, if possible.

        Besides of the network size (e.g. number of nodes), each
        network model consistes of the following set of parameters:
        - distribution parameter (e.g. rate)
        - response window size (e.g. r)
        - coupling strength (e.g. c)
        - density (e.g. rho)

        However, a network model does only require 3 degrees of freedom,
        so the abovementioned set is redundant. This function completes
        the missing parameter if it is missing, or checks if the given
        set is consistent with the model constraints.
        """

        params = [self.rate, self.response, self.coupling, self.density]
        nb_params = sum(1 for _ in params if _ is not None)

        if nb_params == 4:
            if verbose:
                print(params)
            eps = governing_equation(
                density=self.density,
                coupling=self.coupling,
                response=self.response,
                rate=self.rate,
            )
            if eps > 1e-6:
                raise AttributeError(
                    f"Parameters are inconsistent with model constraints. "
                    f"Equation is not satisfied: {eps}"
                )
        elif nb_params < 3:
            raise AttributeError(
                "Not enough parameters given to infer " f"redundant ones. {nb_params}"
            )
        else:
            if self.rate is None:
                self.rate = infer_rate_parameter(
                    density=self.density, coupling=self.coupling, response=self.response
                )
            if self.response is None:
                self.response = infer_response_parameter(
                    density=self.density, coupling=self.coupling, rate=self.rate
                )
            if self.coupling is None:
                self.coupling = infer_coupling_parameter(
                    density=self.density, response=self.response, rate=self.rate
                )
            if self.density is None:
                self.density = infer_density_parameter(
                    response=self.response, coupling=self.coupling, rate=self.rate
                )
