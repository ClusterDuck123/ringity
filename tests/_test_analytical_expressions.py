import numpy as np
import pytest

from ringity.networkmodel._analytical_expressions import (
    _local_clustering_integral_classic,
    _local_clustering_integral_noconstants,
    _local_clustering_integral_xy_symmetry,
    _local_clustering_integral_transformed,
    _local_clustering_integral_dbltransformed,
)

# Default integration precision parameters
DEFAULT_EPSABS = 1.49e-05
DEFAULT_EPSREL = 1.49e-05
DEFAULT_NENSEMBLE = 2**3


@pytest.fixture
def betas():
    return np.random.uniform(size=DEFAULT_NENSEMBLE)


@pytest.fixture
def rs():
    return np.random.uniform(0, 0.5, size=DEFAULT_NENSEMBLE)


@pytest.fixture
def cs():
    return np.random.uniform(size=DEFAULT_NENSEMBLE)


@pytest.fixture
def thetas_arr():
    n_thetas = 2**1
    return np.random.uniform(0, 2 * np.pi, size=[DEFAULT_NENSEMBLE, n_thetas])


class TestLocalClusteringCoefficient:
    def test_noconstants(self, betas, rs, cs, thetas_arr):
        for beta, r, c, thetas in zip(betas, rs, cs, thetas_arr):
            for theta_i in thetas:
                I1 = _local_clustering_integral_classic(
                    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
                )
                I2 = _local_clustering_integral_noconstants(
                    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
                )
                assert I1 == pytest.approx(I2, abs=1e-4)

    def test_xy_symmetry(self, betas, rs, cs, thetas_arr):
        for beta, r, c, thetas in zip(betas, rs, cs, thetas_arr):
            for theta_i in thetas:
                I1 = _local_clustering_integral_noconstants(
                    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
                )
                I2 = _local_clustering_integral_xy_symmetry(
                    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
                )
                assert I1 == pytest.approx(I2, abs=1e-4)

    def test_single_transformation(self, betas, rs, cs, thetas_arr):
        for beta, r, c, thetas in zip(betas, rs, cs, thetas_arr):
            for theta_i in thetas:
                I1 = _local_clustering_integral_xy_symmetry(
                    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
                )
                I2 = _local_clustering_integral_transformed(
                    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
                )
                assert I1 == pytest.approx(I2, abs=1e-4)

    def test_double_transformation(self, betas, rs, cs, thetas_arr):
        for beta, r, c, thetas in zip(betas, rs, cs, thetas_arr):
            for theta_i in thetas:
                I1 = _local_clustering_integral_transformed(
                    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
                )
                I2 = _local_clustering_integral_dbltransformed(
                    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
                )
                assert I1 == pytest.approx(I2, abs=1e-4)
