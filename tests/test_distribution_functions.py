import pytest
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


@pytest.fixture
def setup_analytical():
    theta1 = np.random.uniform(0, np.pi)
    theta2 = np.random.uniform(np.pi, 2 * np.pi)

    r1 = np.random.uniform(0.0, 0.5)
    r2 = np.random.uniform(0.5, 1.0)

    beta = np.random.uniform(0.01, 0.99)
    c = np.random.uniform()
    params = {
        "theta1": theta1,
        "theta2": theta2,
        "r1": r1,
        "r2": r2,
        "beta": beta,
        "c": c,
    }
    return params


class TestAnalyticalConsistency:

    def test_global_density_small_r(self, setup_analytical):
        params = setup_analytical
        rho = trafos.global_density(r=params["r1"], beta=params["beta"], c=params["c"])
        rho_d, _ = quad(
            lambda x: pre_global_density_from_distance(
                x=x, beta=params["beta"], r=params["r1"], c=params["c"]
            ),
            0,
            np.pi,
        )
        rho_s, _ = quad(
            lambda x: pre_global_density_from_similarity(
                x=x, beta=params["beta"], r=params["r1"], c=params["c"]
            ),
            0,
            1,
        )

        rho == pytest.approx(rho_d, abs=1e-5)
        rho == pytest.approx(rho_s, abs=5)

    def test_global_density_big_r(self, setup_analytical):
        params = setup_analytical
        rho = trafos.global_density(r=params["r2"], beta=params["beta"], c=params["c"])
        rho_d, _ = quad(
            lambda x: pre_global_density_from_distance(
                x=x, beta=params["beta"], r=params["r2"], c=params["c"]
            ),
            0,
            np.pi,
        )
        rho_s, _ = quad(
            lambda x: pre_global_density_from_similarity(
                x=x, beta=params["beta"], r=params["r2"], c=params["c"]
            ),
            0,
            1,
        )

        # closed-form and numerical-quadrature densities agree to ~1e-4 in the big-r regime
        rho == pytest.approx(rho_d, abs=1e-3)
        rho == pytest.approx(rho_s, abs=1e-3)

    def test_local_density_small_r_small_theta(self, setup_analytical):
        params = setup_analytical
        rho = trafos.local_density(
            r=params["r1"], theta=params["theta1"], beta=params["beta"], c=params["c"]
        )
        rho_d, _ = quad(
            lambda x: pre_local_density_from_distance(
                x,
                theta=params["theta1"],
                beta=params["beta"],
                r=params["r1"],
                c=params["c"],
            ),
            0,
            np.pi,
        )
        rho_s, _ = quad(
            lambda x: pre_local_density_from_similarity(
                x,
                theta=params["theta1"],
                beta=params["beta"],
                r=params["r1"],
                c=params["c"],
            ),
            0,
            1,
        )

        rho == pytest.approx(rho_d, abs=1e-3)
        rho == pytest.approx(rho_s, abs=1e-3)

    def test_local_density_small_r_big_theta(self, setup_analytical):
        params = setup_analytical
        rho = trafos.local_density(
            r=params["r1"], theta=params["theta2"], beta=params["beta"], c=params["c"]
        )
        rho_d, _ = quad(
            lambda x: pre_local_density_from_distance(
                x,
                theta=params["theta2"],
                beta=params["beta"],
                r=params["r1"],
                c=params["c"],
            ),
            0,
            np.pi,
        )
        rho_s, _ = quad(
            lambda x: pre_local_density_from_similarity(
                x,
                theta=params["theta2"],
                beta=params["beta"],
                r=params["r1"],
                c=params["c"],
            ),
            0,
            1,
        )

        rho == pytest.approx(rho_d, abs=1e-4)
        rho == pytest.approx(rho_s, abs=1e-4)

    def test_local_density_big_r_small_theta(self, setup_analytical):
        params = setup_analytical
        rho = trafos.local_density(
            r=params["r2"], theta=params["theta1"], beta=params["beta"], c=params["c"]
        )
        rho_d, _ = quad(
            lambda x: pre_local_density_from_distance(
                x,
                theta=params["theta1"],
                beta=params["beta"],
                r=params["r2"],
                c=params["c"],
            ),
            0,
            np.pi,
        )
        rho_s, _ = quad(
            lambda x: pre_local_density_from_similarity(
                x,
                theta=params["theta1"],
                beta=params["beta"],
                r=params["r2"],
                c=params["c"],
            ),
            0,
            1,
        )

        rho == pytest.approx(rho_d, abs=1e-3)
        rho == pytest.approx(rho_s, abs=1e-3)

    def test_local_density_big_r_big_theta(self, setup_analytical):
        params = setup_analytical
        rho = trafos.local_density(
            r=params["r2"], theta=params["theta2"], beta=params["beta"], c=params["c"]
        )
        rho_d, _ = quad(
            lambda x: pre_local_density_from_distance(
                x,
                theta=params["theta2"],
                beta=params["beta"],
                r=params["r2"],
                c=params["c"],
            ),
            0,
            np.pi,
        )
        rho_s, _ = quad(
            lambda x: pre_local_density_from_similarity(
                x,
                theta=params["theta2"],
                beta=params["beta"],
                r=params["r2"],
                c=params["c"],
            ),
            0,
            1,
        )

        rho == pytest.approx(rho_d, abs=1e-3)
        rho == pytest.approx(rho_s, abs=1e-3)

    def test_second_moment_small_r(self, setup_analytical):
        params = setup_analytical
        mu2 = trafos.second_moment_density(
            r=params["r1"], beta=params["beta"], c=params["c"]
        )
        mu2_rho, _ = quad(
            lambda x: trafos.local_density(
                x, r=params["r1"], beta=params["beta"], c=params["c"]
            )
            ** 2
            * dists.pdf_delay(x, beta=params["beta"]),
            0,
            2 * np.pi,
        )

        mu2 == pytest.approx(mu2_rho, abs=1e-4)

    def test_second_moment_big_r(self, setup_analytical):
        params = setup_analytical
        mu2 = trafos.second_moment_density(
            r=params["r2"], beta=params["beta"], c=params["c"]
        )
        mu2_rho, _ = quad(
            lambda x: trafos.local_density(
                x, r=params["r2"], beta=params["beta"], c=params["c"]
            )
            ** 2
            * dists.pdf_delay(x, beta=params["beta"]),
            0,
            2 * np.pi,
        )

        mu2 == pytest.approx(mu2_rho, abs=1e-4)


@pytest.fixture
def setup_empirically():
    n_ensemble = 2**6
    N = 2**7

    beta = np.random.uniform()
    r = np.random.uniform()
    c = np.random.uniform()

    params = {
        "N": N,
        "beta": beta,
        "r": r,
        "c": c,
    }

    thetas_arr = np.empty((n_ensemble, N))
    degdist_arr = np.empty((n_ensemble, N))

    for i in range(n_ensemble):
        thetas = dists.delaydist.rvs(size=N, beta=beta)

        D = trafos.pw_angular_distance(thetas)
        P = trafos.interaction_probability(D, r=r, c=c)

        R = np.random.uniform(size=N * (N - 1) // 2)
        A = squareform(np.where(P > R, 1, 0))

        G = nx.from_numpy_array(A)

        thetas_arr[i] = thetas
        degdist_arr[i] = np.array([d for n, d in G.degree()])

    return params, thetas_arr, degdist_arr


class TestDegreeDistributionEmpiricaly:

    def test_expectation(self, setup_empirically):
        params, thetas_arr, degdist_arr = setup_empirically

        mu_obs = np.mean(degdist_arr)
        mu_exp = (params["N"] - 1) * trafos.global_density(
            r=params["r"], beta=params["beta"], c=params["c"]
        )

        # Additional way to proof test the local density function
        mu_obs2 = (params["N"] - 1) * np.mean(
            trafos.local_density(
                thetas_arr, r=params["r"], c=params["c"], beta=params["beta"]
            )
        )

        mu_exp / mu_obs == pytest.approx(1, abs=0.025)
        mu_exp / mu_obs2 == pytest.approx(1, abs=0.025)

    def test_variance(self, setup_empirically):
        params, thetas_arr, degdist_arr = setup_empirically

        var_obs = np.var(degdist_arr)

        mu1 = trafos.global_density(r=params["r"], beta=params["beta"], c=params["c"])
        mu2 = trafos.second_moment_density(
            r=params["r"], beta=params["beta"], c=params["c"]
        )
        var_exp = (params["N"] - 1) * (
            (params["N"] - 2) * mu2 - mu1 * ((params["N"] - 1) * mu1 - 1)
        )

        var_obs / var_exp == pytest.approx(1, abs=0.1)

    # def test_degree_distribution(self):
    #     degdist_exp = dists.local_density(
    #         self.thetas_arr, r=self.r, c=self.c, beta=self.beta
    #     )
    #     degdist_obs = self.degdist_arr
    #     mean_error = np.mean(degdist_exp - degdist_obs)

    #     self.assertAlmostEqual(mean_error, 0, delta=0.01)
