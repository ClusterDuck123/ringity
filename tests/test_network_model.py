import unittest
import numpy as np
import ringity as rng
import networkx as nx

from ringity.networkmodel.param_utils import (
    infer_density_parameter,
    infer_rate_parameter,
    infer_response_parameter,
    infer_coupling_parameter,
)
from ringity.networkmodel.transformations import beta_to_rate


class TestHardCodedNetworkModel(unittest.TestCase):
    def test_small_r(self):
        random_state = 240322

        N = 2**6
        r = 0.123
        beta = 0.234
        c = 2 * 0.345

        G = rng.network_model(
            N=N,
            response=r,
            beta=beta,
            coupling=c,
            random_state=random_state,
        )

        hard_score = 0.6138893378353827
        obs_score = rng.ring_score(G)

        self.assertAlmostEqual(hard_score, obs_score)

    def test_big_r(self):
        random_state = 240322

        N = 2**6
        r = 2 * 0.345
        beta = 0.234
        c = 0.123

        G = rng.network_model(
            N=N,
            response=r,
            beta=beta,
            coupling=c,
            random_state=random_state,
        )

        hard_score = 0.11653461783805241
        obs_score = rng.ring_score(G)

        self.assertAlmostEqual(hard_score, obs_score)


class TestExponentialNetworkModel(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.randint(2**10)
        self.N = 2**7

        self.beta = np.random.uniform()
        self.response = np.random.uniform()
        self.coupling = np.random.uniform()

        self.rate = beta_to_rate(self.beta)
        self.density = infer_density_parameter(
            rate=self.rate,
            response=self.response,
            coupling=self.coupling,
        )

        self.G = rng.network_model(
            N=self.N,
            response=self.response,
            beta=self.beta,
            coupling=self.coupling,
            random_state=self.random_state,
        )

    def test_response_parameter_consistency(self):
        G_r = rng.network_model(
            N=self.N,
            r=self.response,
            beta=self.beta,
            c=self.coupling,
            random_state=self.random_state,
        )

        G_a = rng.network_model(
            N=self.N,
            a=self.response,
            beta=self.beta,
            c=self.coupling,
            random_state=self.random_state,
        )

        G_alpha = rng.network_model(
            N=self.N,
            alpha=self.response,
            beta=self.beta,
            c=self.coupling,
            random_state=self.random_state,
        )

        self.assertTrue(nx.is_isomorphic(self.G, G_r))
        self.assertTrue(nx.is_isomorphic(self.G, G_a))
        self.assertTrue(nx.is_isomorphic(self.G, G_alpha))

    def test_coupling_parameter_consistency(self):
        G_c = rng.network_model(
            N=self.N,
            alpha=self.response,
            beta=self.beta,
            c=self.coupling,
            random_state=self.random_state,
        )

        G_K = rng.network_model(
            N=self.N,
            alpha=self.response,
            beta=self.beta,
            K=self.coupling,
            random_state=self.random_state,
        )
        self.assertTrue(nx.is_isomorphic(self.G, G_c))
        self.assertTrue(nx.is_isomorphic(self.G, G_K))

    def test_density_parameter_consistency(self):
        G_density = rng.network_model(
            N=self.N,
            alpha=self.response,
            beta=self.beta,
            density=self.density,
            random_state=self.random_state,
        )

        G_rho = rng.network_model(
            N=self.N,
            alpha=self.response,
            beta=self.beta,
            rho=self.density,
            random_state=self.random_state,
        )
        self.assertTrue(nx.is_isomorphic(G_density, G_rho))

    def test_density_calculation(self):
        G_gen = (
            rng.network_model(
                N=self.N,
                response=self.response,
                beta=self.beta,
                coupling=self.coupling,
            )
            for _ in range(2**5)
        )

        mean_density = np.mean(list(map(nx.density, G_gen)))

        self.assertTrue(np.isclose(mean_density, self.density, rtol=1e-01))

    def test_parameter_inference(self):
        response = infer_response_parameter(
            rate=self.rate, coupling=self.coupling, density=self.density
        )

        coupling = infer_coupling_parameter(
            rate=self.rate, response=self.response, density=self.density
        )

        rate = infer_rate_parameter(
            response=self.response, coupling=self.coupling, density=self.density
        )

        self.assertAlmostEqual(rate, self.rate)
        self.assertAlmostEqual(response, self.response)
        self.assertAlmostEqual(coupling, self.coupling)


if __name__ == "__main__":
    unittest.main()
