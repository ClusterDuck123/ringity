import numpy as np
import ringity.networkmodel.distributions as dists
import ringity.networkmodel.transformations as trafos
import unittest

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


class TestLocalClusteringCoefficient(unittest.TestCase):
    def setUp(self):
        n_ensemble = DEFAULT_NENSEMBLE
        n_thetas = 2**1

        self.betas = np.random.uniform(size=n_ensemble)
        self.rs = np.random.uniform(0, 0.5, size=n_ensemble)
        self.cs = np.random.uniform(size=n_ensemble)

        self.thetas = np.random.uniform(0, 2 * np.pi, size=[n_ensemble, n_thetas])

    def test_noconstants(self):
        for beta, r, c, thetas in zip(self.betas, self.rs, self.cs, self.thetas):
            for theta_i in thetas:
                I1 = _local_clustering_integral_classic(
                    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
                )
                I2 = _local_clustering_integral_noconstants(
                    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
                )
                self.assertAlmostEqual(I1, I2, delta=1e-4)

    def test_xy_symmetry(self):
        for beta, r, c, thetas in zip(self.betas, self.rs, self.cs, self.thetas):
            for theta_i in thetas:
                I1 = _local_clustering_integral_noconstants(
                    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
                )
                I2 = _local_clustering_integral_xy_symmetry(
                    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
                )
                self.assertAlmostEqual(I1, I2, delta=1e-4)

    def test_single_transformation(self):
        for beta, r, c, thetas in zip(self.betas, self.rs, self.cs, self.thetas):
            for theta_i in thetas:
                I1 = _local_clustering_integral_xy_symmetry(
                    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
                )
                I2 = _local_clustering_integral_transformed(
                    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
                )
                self.assertAlmostEqual(I1, I2, delta=1e-4)

    def test_double_transformation(self):
        for beta, r, c, thetas in zip(self.betas, self.rs, self.cs, self.thetas):
            for theta_i in thetas:
                I1 = _local_clustering_integral_transformed(
                    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
                )
                I2 = _local_clustering_integral_dbltransformed(
                    theta_i, r, c, beta, epsabs=DEFAULT_EPSABS, epsrel=DEFAULT_EPSREL
                )
                self.assertAlmostEqual(I1, I2, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
