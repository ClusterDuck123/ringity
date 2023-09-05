import unittest
import numpy as np
import ringity as rng
import networkx as nx

from scipy.spatial.distance import pdist, squareform

class TestPointCloud(unittest.TestCase):
    def setUp(self):
        self.X = np.random.uniform(size = [2**5, 3])
        self.D = squareform(pdist(self.X))
        self.G = nx.erdos_renyi_graph(2**5, p = 0.3)

    def test_point_cloud_downstreams(self):
        pdgm_X_gen = rng.pdiagram(self.X)
        pdgm_X_spe = rng.pdiagram_from_point_cloud(self.X)
        
        pdgm_D_gen = rng.pdiagram(self.D)
        pdgm_D_spe = rng.pdiagram_from_distance_matrix(self.D)
        
        self.assertTrue(pdgm_X_gen == pdgm_X_spe)
        self.assertTrue(pdgm_X_spe == pdgm_D_gen)
        self.assertTrue(pdgm_D_gen == pdgm_D_spe)

        score_X_gen = rng.ring_score(self.X)
        score_X_spe = rng.ring_score_from_point_cloud(self.X)

        score_D_gen = rng.ring_score(self.D)
        score_D_spe = rng.ring_score_from_distance_matrix(self.D)

        score_pdgm_gen = rng.ring_score(pdgm_X_gen)
        score_pdgm_spe = rng.ring_score_from_pdiagram(pdgm_X_gen)

        self.assertTrue(score_X_gen == score_X_spe)
        self.assertTrue(score_X_spe == score_D_gen)
        self.assertTrue(score_D_gen == score_D_spe)
        self.assertTrue(score_D_spe == score_pdgm_gen)
        self.assertTrue(score_pdgm_gen == score_pdgm_spe)

    def test_network_downstreams(self):
        pdgm_G_gen = rng.pdiagram(self.G)
        pdgm_G_spe = rng.pdiagram_from_network(self.G)
        pdgm_G_net_gen = rng.pdiagram(self.G, metric = 'net_flow')
        pdgm_G_net_spe = rng.pdiagram_from_network(self.G, metric = 'net_flow')

        self.assertTrue(pdgm_G_gen == pdgm_G_spe)
        self.assertTrue(pdgm_G_spe == pdgm_G_net_gen)
        self.assertTrue(pdgm_G_net_gen == pdgm_G_net_spe)

        score_G_gen = rng.ring_score(self.G)
        score_G_spe = rng.ring_score_from_network(self.G)
        score_G_net_gen = rng.ring_score(self.G, metric = 'net_flow')
        score_G_net_spe = rng.ring_score_from_network(self.G, metric = 'net_flow')

        self.assertTrue(score_G_gen == score_G_spe)
        self.assertTrue(score_G_spe == score_G_net_gen)
        self.assertTrue(score_G_net_gen == score_G_net_spe)
    
    def test_anndata(self):
        # adata = anndata.AnnData(self.X)
        # pdgm_adata_gen = rng.pdiagram(adata)
        # pdgm_adata_spe = rng.pdiagram_from_AnnData(adata)

        # score_adata_gen = rng.ring_score(adata)
        # score_adata_spe = rng.ring_score_from_AnnData(adata)
        pass


if __name__ == '__main__':
    unittest.main()
