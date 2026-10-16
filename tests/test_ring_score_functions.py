import pytest
import numpy as np
import ringity as rty
import networkx as nx

from scipy.spatial.distance import pdist, squareform


@pytest.fixture
def setup():
    X = np.random.uniform(size=[2**5, 3])
    D = squareform(pdist(X))
    G = nx.erdos_renyi_graph(2**5, p=0.3)
    return X, D, G


class TestPointCloud:
    def test_point_cloud_downstreams(self, setup):
        X, D, G = setup
        pdgm_X_gen = rty.pdiagram(X)
        pdgm_X_spe = rty.pdiagram_from_point_cloud(X)

        pdgm_D_gen = rty.pdiagram(D)
        pdgm_D_spe = rty.pdiagram_from_distance_matrix(D)

        assert pdgm_X_gen == pdgm_X_spe
        assert pdgm_X_spe == pdgm_D_gen
        assert pdgm_D_gen == pdgm_D_spe

        score_X_gen = rty.ring_score(X)
        score_X_spe = rty.ring_score_from_point_cloud(X)

        score_D_gen = rty.ring_score(D)
        score_D_spe = rty.ring_score_from_distance_matrix(D)

        score_pdgm_gen = rty.ring_score(pdgm_X_gen)
        score_pdgm_spe = rty.ring_score_from_pdiagram(pdgm_X_gen)

        assert score_X_gen == score_X_spe
        assert score_X_spe == score_D_gen
        assert score_D_gen == score_D_spe
        assert score_D_spe == score_pdgm_gen
        assert score_pdgm_gen == score_pdgm_spe

    def test_network_downstreams(self, setup):
        X, D, G = setup

        pdgm_G_gen = rty.pdiagram(G)
        pdgm_G_spe = rty.pdiagram_from_network(G)
        pdgm_G_net_gen = rty.pdiagram(G, metric="net_flow")
        pdgm_G_net_spe = rty.pdiagram_from_network(G, metric="net_flow")

        assert pdgm_G_gen == pdgm_G_spe
        assert pdgm_G_spe == pdgm_G_net_gen
        assert pdgm_G_net_gen == pdgm_G_net_spe

        score_G_gen = rty.ring_score(G)
        score_G_spe = rty.ring_score_from_network(G)
        score_G_net_gen = rty.ring_score(G, metric="net_flow")
        score_G_net_spe = rty.ring_score_from_network(G, metric="net_flow")

        assert score_G_gen == score_G_spe
        assert score_G_spe == score_G_net_gen
        assert score_G_net_gen == score_G_net_spe

    def test_anndata(self):
        # adata = anndata.AnnData(self.X)
        # pdgm_adata_gen = rty.pdiagram(adata)
        # pdgm_adata_spe = rty.pdiagram_from_AnnData(adata)

        # score_adata_gen = rty.ring_score(adata)
        # score_adata_spe = rty.ring_score_from_AnnData(adata)
        pass
