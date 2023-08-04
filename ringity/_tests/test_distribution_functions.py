import unittest
import numpy as np
import networkx as nx
import ringity.networks.networkmodel.distributions as dists
import ringity.networks.networkmodel.transformations as trafos

from scipy.integrate import quad
from scipy.spatial.distance import squareform


def pre_global_density_from_distance(x, beta, r, c):
    p = trafos.interaction_probability(x, r=r, c=c)
    f_D = dists.pdf_distance(x, beta=beta)
    return p * f_D


def pre_global_density_from_similarity(x, beta, r, c):
    F_S = dists.cdf_similarity(x, r=r, beta=beta)
    return c * (1 - F_S)


def pre_local_density_from_distance(x, theta, beta, r, c):
    p = trafos.interaction_probability(x, r=r, c=c)
    f_D = dists.pdf_conditional_distance(x, theta=theta, beta=beta)
    return p * f_D


def pre_local_density_from_similarity(x, theta, beta, r, c):
    F_S = dists.cdf_conditional_similarity(x, theta=theta, r=r, beta=beta)
    return c * (1 - F_S)


class TestAnalyticalConsistency(unittest.TestCase):
    def setUp(self):
        self.theta1 = np.random.uniform(0, np.pi)
        self.theta2 = np.random.uniform(np.pi, 2 * np.pi)

        self.r1 = np.random.uniform(0.0, 0.5)
        self.r2 = np.random.uniform(0.5, 1.0)

        self.beta = np.random.uniform()
        self.c = np.random.uniform()

    def test_global_density_r_small(self):
        rho = dists.global_density(r=self.r1, beta=self.beta, c=self.c)
        rho_d, _ = quad(
            lambda x: pre_global_density_from_distance(
                x=x, beta=self.beta, r=self.r1, c=self.c
            ),
            0,
            np.pi,
        )
        rho_s, _ = quad(
            lambda x: pre_global_density_from_similarity(
                x=x, beta=self.beta, r=self.r1, c=self.c
            ),
            0,
            1,
        )

        self.assertAlmostEqual(rho, rho_d, places=5)
        self.assertAlmostEqual(rho, rho_s, places=5)

    def test_global_density_r_big(self):
        rho = dists.global_density(r=self.r2, beta=self.beta, c=self.c)
        rho_d, _ = quad(
            lambda x: pre_global_density_from_distance(
                x=x, beta=self.beta, r=self.r2, c=self.c
            ),
            0,
            np.pi,
        )
        rho_s, _ = quad(
            lambda x: pre_global_density_from_similarity(
                x=x, beta=self.beta, r=self.r2, c=self.c
            ),
            0,
            1,
        )

        self.assertAlmostEqual(rho, rho_d, places=5)
        self.assertAlmostEqual(rho, rho_s, places=5)

    def test_local_density_r_small_theta_small(self):
        rho = dists.local_density(
            r=self.r1, theta=self.theta1, beta=self.beta, c=self.c
        )
        rho_d, _ = quad(
            lambda x: pre_local_density_from_distance(
                x, theta=self.theta1, beta=self.beta, r=self.r1, c=self.c
            ),
            0,
            np.pi,
        )
        rho_s, _ = quad(
            lambda x: pre_local_density_from_similarity(
                x, theta=self.theta1, beta=self.beta, r=self.r1, c=self.c
            ),
            0,
            1,
        )

        self.assertAlmostEqual(rho, rho_d, places=5)
        self.assertAlmostEqual(rho, rho_s, places=5)

    def test_local_density_r_small_theta_big(self):
        rho = dists.local_density(
            r=self.r1, theta=self.theta2, beta=self.beta, c=self.c
        )
        rho_d, _ = quad(
            lambda x: pre_local_density_from_distance(
                x, theta=self.theta2, beta=self.beta, r=self.r1, c=self.c
            ),
            0,
            np.pi,
        )
        rho_s, _ = quad(
            lambda x: pre_local_density_from_similarity(
                x, theta=self.theta2, beta=self.beta, r=self.r1, c=self.c
            ),
            0,
            1,
        )

        self.assertAlmostEqual(rho, rho_d, places=5)
        self.assertAlmostEqual(rho, rho_s, places=5)

    def test_local_density_r_big_theta_small(self):
        rho1 = dists.local_density(
            r=self.r2, theta=self.theta1, beta=self.beta, c=self.c
        )
        rho1_d, _ = quad(
            lambda x: pre_local_density_from_distance(
                x, theta=self.theta1, beta=self.beta, r=self.r2, c=self.c
            ),
            0,
            np.pi,
        )
        rho1_s, _ = quad(
            lambda x: pre_local_density_from_similarity(
                x, theta=self.theta1, beta=self.beta, r=self.r2, c=self.c
            ),
            0,
            1,
        )

        self.assertAlmostEqual(rho1, rho1_d, places=5)
        self.assertAlmostEqual(rho1, rho1_s, places=5)

    def test_local_density_r_big_theta_big(self):
        rho2 = dists.local_density(
            r=self.r2, theta=self.theta2, beta=self.beta, c=self.c
        )
        rho2_d, _ = quad(
            lambda x: pre_local_density_from_distance(
                x, theta=self.theta2, beta=self.beta, r=self.r2, c=self.c
            ),
            0,
            np.pi,
        )
        rho2_s, _ = quad(
            lambda x: pre_local_density_from_similarity(
                x, theta=self.theta2, beta=self.beta, r=self.r2, c=self.c
            ),
            0,
            1,
        )

        self.assertAlmostEqual(rho2, rho2_d, places=5)
        self.assertAlmostEqual(rho2, rho2_s, places=5)


class TestEmpiricalConsistency(unittest.TestCase):
    def setUp(self):
        self.N = 2**8

        self.beta = np.random.uniform()
        self.r = np.random.uniform()
        self.c = np.random.uniform()

        self.thetas = dists.delaydist.rvs(size=self.N, beta=self.beta)

        self.G_list = []

        for _ in range(2**3):
            D = trafos.pw_circular_distance(self.thetas)
            P = trafos.interaction_probability(D, r=self.r, c=self.c)

            R = np.random.uniform(size=self.N*(self.N-1)//2)
            A = squareform(np.where(P > R, 1, 0))
            
            self.G_list.append(nx.from_numpy_array(A))

    def test_global_density(self):
        rho_exp = dists.global_density(r=self.r, beta=self.beta, c=self.c)
        rho_obs = np.mean([nx.density(G) for G in self.G_list])

        self.assertAlmostEqual(rho_exp, rho_obs, places=1)


if __name__ == "__main__":
    unittest.main()
