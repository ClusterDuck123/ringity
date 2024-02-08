import unittest
import numpy as np
import networkx as nx
import ringity.networkmodel.distributions as dists
import ringity.networkmodel.transformations as trafos

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

        self.beta = np.random.uniform(0.01, 0.99)
        self.c = np.random.uniform()

    def test_global_density_small_r(self):
        rho = trafos.global_density(r=self.r1, beta=self.beta, c=self.c)
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

    def test_global_density_big_r(self):
        rho = trafos.global_density(r=self.r2, beta=self.beta, c=self.c)
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

    def test_local_density_small_r_small_theta(self):
        rho = trafos.local_density(
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

        self.assertAlmostEqual(rho, rho_d, places=3)
        self.assertAlmostEqual(rho, rho_s, places=3)

    def test_local_density_small_r_big_theta(self):
        rho = trafos.local_density(
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

        self.assertAlmostEqual(rho, rho_d, places=4)
        self.assertAlmostEqual(rho, rho_s, places=4)

    def test_local_density_big_r_small_theta(self):
        rho = trafos.local_density(
            r=self.r2, theta=self.theta1, beta=self.beta, c=self.c
        )
        rho_d, _ = quad(
            lambda x: pre_local_density_from_distance(
                x, theta=self.theta1, beta=self.beta, r=self.r2, c=self.c
            ),
            0,
            np.pi,
        )
        rho_s, _ = quad(
            lambda x: pre_local_density_from_similarity(
                x, theta=self.theta1, beta=self.beta, r=self.r2, c=self.c
            ),
            0,
            1,
        )

        self.assertAlmostEqual(rho, rho_d, places=3)
        self.assertAlmostEqual(rho, rho_s, places=3)

    def test_local_density_big_r_big_theta(self):
        rho = trafos.local_density(
            r=self.r2, theta=self.theta2, beta=self.beta, c=self.c
        )
        rho_d, _ = quad(
            lambda x: pre_local_density_from_distance(
                x, theta=self.theta2, beta=self.beta, r=self.r2, c=self.c
            ),
            0,
            np.pi,
        )
        rho_s, _ = quad(
            lambda x: pre_local_density_from_similarity(
                x, theta=self.theta2, beta=self.beta, r=self.r2, c=self.c
            ),
            0,
            1,
        )

        self.assertAlmostEqual(rho, rho_d, places=3)
        self.assertAlmostEqual(rho, rho_s, places=3)

    def test_second_moment_small_r(self):
        mu2 = trafos.second_moment_density(r=self.r1, beta=self.beta, c=self.c)
        mu2_rho, _ = quad(
            lambda x: trafos.local_density(x, r=self.r1, beta=self.beta, c=self.c) ** 2
            * dists.pdf_delay(x, beta=self.beta),
            0,
            2 * np.pi,
        )

        self.assertAlmostEqual(mu2, mu2_rho, places=4)

    def test_second_moment_big_r(self):
        mu2 = trafos.second_moment_density(r=self.r2, beta=self.beta, c=self.c)
        mu2_rho, _ = quad(
            lambda x: trafos.local_density(x, r=self.r2, beta=self.beta, c=self.c) ** 2
            * dists.pdf_delay(x, beta=self.beta),
            0,
            2 * np.pi,
        )

        self.assertAlmostEqual(mu2, mu2_rho, places=4)


class TestDegreeDistributionEmpiricaly(unittest.TestCase):
    def setUp(self):
        n_ensemble = 2**6
        self.N = 2**7

        self.beta = np.random.uniform()
        self.r = np.random.uniform()
        self.c = np.random.uniform()

        self.thetas_arr = np.empty((n_ensemble, self.N))
        self.degdist_arr = np.empty((n_ensemble, self.N))

        for i in range(n_ensemble):
            thetas = dists.delaydist.rvs(size=self.N, beta=self.beta)

            D = trafos.pw_angular_distance(thetas)
            P = trafos.interaction_probability(D, r=self.r, c=self.c)

            R = np.random.uniform(size=self.N * (self.N - 1) // 2)
            A = squareform(np.where(P > R, 1, 0))

            G = nx.from_numpy_array(A)

            self.thetas_arr[i] = thetas
            self.degdist_arr[i] = np.array([d for n, d in G.degree()])

    def test_expectation(self):
        mu_obs = np.mean(self.degdist_arr)
        mu_exp = (self.N - 1) * trafos.global_density(
            r=self.r, beta=self.beta, c=self.c
        )

        # Additional way to proof test the local density function
        mu_obs2 = (self.N - 1) * np.mean(
            trafos.local_density(self.thetas_arr, r=self.r, c=self.c, beta=self.beta)
        )

        self.assertAlmostEqual(mu_exp / mu_obs, 1, delta=0.025)
        self.assertAlmostEqual(mu_exp / mu_obs2, 1, delta=0.025)

    def test_variance(self):
        var_obs = np.var(self.degdist_arr)

        mu1 = trafos.global_density(r=self.r, beta=self.beta, c=self.c)
        mu2 = trafos.second_moment_density(r=self.r, beta=self.beta, c=self.c)
        var_exp = (self.N - 1) * ((self.N - 2) * mu2 - mu1 * ((self.N - 1) * mu1 - 1))

        self.assertAlmostEqual(var_obs / var_exp, 1, delta=0.075)

    # def test_degree_distribution(self):
    #     degdist_exp = dists.local_density(
    #         self.thetas_arr, r=self.r, c=self.c, beta=self.beta
    #     )
    #     degdist_obs = self.degdist_arr
    #     mean_error = np.mean(degdist_exp - degdist_obs)

    #     self.assertAlmostEqual(mean_error, 0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
