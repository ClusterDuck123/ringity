import unittest
import numpy as np
import ringity as rng
import networkx as nx

class TestExponentialNetworkModel(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.randint(2**10)
        self.N = 2**7

        self.beta = np.random.uniform()
        self.r = np.random.uniform()
        self.c = np.random.uniform()

        self.a = self.r
        self.alpha = self.r
        self.K = self.c

        self.G = rng.network_model(N = self.N,
                                  r = self.r ,
                                  beta = self.beta,
                                  K = self.K,
                                  random_state = self.random_state)

        self.G_a = rng.network_model(N = self.N,
                                     a = self.a ,
                                     beta = self.beta,
                                     K = self.K,
                                     random_state = self.random_state)

        self.G_alpha = rng.network_model(N = self.N,
                                        alpha = self.alpha ,
                                        beta = self.beta,
                                        K = self.K,
                                        random_state = self.random_state)

    def test_response_parameter_consistency(self):
        self.assertTrue(nx.is_isomorphic(self.G, self.G_a))
        self.assertTrue(nx.is_isomorphic(self.G, self.G_alpha))



if __name__ == '__main__':
    unittest.main()
