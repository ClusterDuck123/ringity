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

        self.G = rng.network_model(
                                N = self.N,
                                r = self.r ,
                                beta = self.beta,
                                c = self.c,
                                random_state = self.random_state)

        self.G_a = rng.network_model(
                                N = self.N,
                                a = self.r ,
                                beta = self.beta,
                                c = self.c,
                                random_state = self.random_state)

        self.G_alpha = rng.network_model(
                                N = self.N,
                                alpha = self.r ,
                                beta = self.beta,
                                c = self.c,
                                random_state = self.random_state)
                                
        self.G_K = rng.network_model(
                                N = self.N,
                                alpha = self.r ,
                                beta = self.beta,
                                K = self.c,
                                random_state = self.random_state)

    def test_response_parameter_consistency(self):
        self.assertTrue(nx.is_isomorphic(self.G, self.G_a))
        self.assertTrue(nx.is_isomorphic(self.G, self.G_alpha))
        
    def test_coupling_parameter_consistency(self):
        self.assertTrue(nx.is_isomorphic(self.G, self.G_K))
        
    def test_implicit_parameter_consistency(self):
        # response parameter
        # coupling parameter
        pass



if __name__ == '__main__':
    unittest.main()
