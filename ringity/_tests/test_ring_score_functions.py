import unittest
import numpy as np
import ringity as rng

from scipy.spatial.distance import pdist, squareform

class TestPointCloud(unittest.TestCase):
    def setUp(self):
        self.X = np.random.uniform(size = [2**5, 3])
        self.D = squareform(pdist(self.X))

    def test_from_distance_matrix(self):
        dgm1 = rng.pdiagram(self.X)
        dgm2 = rng.pdiagram_from_point_cloud(self.X)
        
        dgm3 = rng.pdiagram(self.D)
        dgm4 = rng.pdiagram_from_distance_matrix(self.D)
        
        self.assertTrue(dgm1 == dgm2)
        self.assertTrue(dgm2 == dgm3)
        self.assertTrue(dgm3 == dgm4)

        score1 = rng.ring_score(self.X)
        score2 = rng.ring_score_from_point_cloud(self.X)

        score3 = rng.ring_score(self.D)
        score4 = rng.ring_score_from_distance_matrix(self.D)

        score5 = rng.ring_score_from_pdiagram(dgm1)

        self.assertTrue(score1 == score2)
        self.assertTrue(score2 == score3)
        self.assertTrue(score3 == score4)
        self.assertTrue(score4 == score5)


if __name__ == '__main__':
    unittest.main()
