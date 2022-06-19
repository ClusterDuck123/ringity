from ringity.network_models.networkbuilder import NetworkBuilder
from ringity.network_models.param_utils import (
                                get_response_parameter,
                                get_rate_parameter,
                                get_coupling_parameter
                                )


network_builder = NetworkBuilder()

network_builder.N = 1234
network_builder.response = get_response_parameter(a=None, alpha=None, r=0.123)
network_builder.rate = get_rate_parameter(rate = 0.456, beta = None)
