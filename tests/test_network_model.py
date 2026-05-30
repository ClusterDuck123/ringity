import pytest
import ringity as rty
import numpy as np
import networkx as nx

from ringity.networkmodel.param_utils import (
    infer_density_parameter,
    infer_rate_parameter,
    infer_response_parameter,
    infer_coupling_parameter,
)
from ringity.networkmodel.transformations import beta_to_rate


class TestHardCodedNetworkModel:
    def test_small_r(self):
        random_state = 240322

        N = 2**6
        r = 0.123
        beta = 0.234
        c = 2 * 0.345

        G = rty.network_model(
            N=N,
            response=r,
            beta=beta,
            coupling=c,
            random_state=random_state,
        )

        hard_score = 0.6138893378353827
        obs_score = rty.ring_score(G)

        assert hard_score == pytest.approx(obs_score)

    def test_big_r(self):
        random_state = 240322

        N = 2**6
        r = 2 * 0.345
        beta = 0.234
        c = 0.123

        G = rty.network_model(
            N=N,
            response=r,
            beta=beta,
            coupling=c,
            random_state=random_state,
        )

        hard_score = 0.11653461783805241
        obs_score = rty.ring_score(G)

        assert hard_score == pytest.approx(obs_score)


@pytest.fixture
def fixed_network():
    rng = np.random.default_rng()
    random_state = rng.integers(0, 2**10)

    param = {
        "N": 2**7,
        "beta": rng.uniform(),
        "response": rng.uniform(),
        "coupling": rng.uniform(),
    }

    rate = beta_to_rate(param["beta"])
    density = infer_density_parameter(
        rate=rate,
        response=param["response"],
        coupling=param["coupling"],
    )

    param["rate"] = rate
    param["density"] = density

    G = rty.network_model(
        N=param["N"],
        response=param["response"],
        beta=param["beta"],
        coupling=param["coupling"],
        random_state=random_state,
    )
    return G, param, random_state


class TestExponentialNetworkModel:
    # def setUp(self):
    #     self.random_state = np.random.randint(2**10)
    #     self.N = 2**7

    #     self.beta = np.random.uniform()
    #     self.response = np.random.uniform()
    #     self.coupling = np.random.uniform()

    #     self.rate = beta_to_rate(self.beta)
    #     self.density = infer_density_parameter(
    #         rate=self.rate,
    #         response=self.response,
    #         coupling=self.coupling,
    #     )

    #     self.G = rg.network_model(
    #         N=self.N,
    #         response=self.response,
    #         beta=self.beta,
    #         coupling=self.coupling,
    #         random_state=self.random_state,
    #     )

    def test_response_parameter_consistency(self, fixed_network):
        G, param, random_state = fixed_network
        G_r = rty.network_model(
            N=param["N"],
            r=param["response"],
            beta=param["beta"],
            c=param["coupling"],
            random_state=random_state,
        )

        G_a = rty.network_model(
            N=param["N"],
            a=param["response"],
            beta=param["beta"],
            c=param["coupling"],
            random_state=random_state,
        )

        G_alpha = rty.network_model(
            N=param["N"],
            alpha=param["response"],
            beta=param["beta"],
            c=param["coupling"],
            random_state=random_state,
        )

        assert nx.is_isomorphic(G, G_r)
        assert nx.is_isomorphic(G, G_a)
        assert nx.is_isomorphic(G, G_alpha)

    def test_coupling_parameter_consistency(self, fixed_network):
        G, param, random_state = fixed_network
        G_c = rty.network_model(
            N=param["N"],
            alpha=param["response"],
            beta=param["beta"],
            c=param["coupling"],
            random_state=random_state,
        )

        G_K = rty.network_model(
            N=param["N"],
            alpha=param["response"],
            beta=param["beta"],
            K=param["coupling"],
            random_state=random_state,
        )
        assert nx.is_isomorphic(G, G_c)
        assert nx.is_isomorphic(G, G_K)

    def test_density_parameter_consistency(self, fixed_network):
        G, param, random_state = fixed_network
        G_density = rty.network_model(
            N=param["N"],
            alpha=param["response"],
            beta=param["beta"],
            density=param["density"],
            random_state=random_state,
        )

        G_rho = rty.network_model(
            N=param["N"],
            alpha=param["response"],
            beta=param["beta"],
            rho=param["density"],
            random_state=random_state,
        )
        assert nx.is_isomorphic(G_density, G_rho)

    def test_density_calculation(self, fixed_network):
        G, param, random_state = fixed_network
        G_gen = (
            rty.network_model(
                N=param["N"],
                response=param["response"],
                beta=param["beta"],
                coupling=param["coupling"],
            )
            for _ in range(2**5)
        )

        mean_density = np.mean(list(map(nx.density, G_gen)))

        assert mean_density == pytest.approx(param["density"], rel=1e-01)

    def test_parameter_inference(self, fixed_network):
        G, param, random_state = fixed_network
        response = infer_response_parameter(
            rate=param["rate"], coupling=param["coupling"], density=param["density"]
        )

        coupling = infer_coupling_parameter(
            rate=param["rate"], response=response, density=param["density"]
        )

        rate = infer_rate_parameter(
            response=response, coupling=coupling, density=param["density"]
        )

        assert rate == pytest.approx(param["rate"])
        assert response == pytest.approx(param["response"])
        assert coupling == pytest.approx(param["coupling"])
