import pytest
import networkx as nx
import ringity as rty
import ringity.networks.centralities as cents
import ringity.tda.pdiagram as pdgm

from pathlib import Path

DIRNAME_TEST_DATA = Path(__file__).parent / "test_data"

FNAME_PDGM = DIRNAME_TEST_DATA / "lipid_coexp_dgm.txt"
FNAME_NETWORK = DIRNAME_TEST_DATA / "lipid_coexp_network.txt"

"""THIS MODULE NEEDS TO BE UPDATED!!"""


@pytest.fixture(scope="session")
def lipid_data():
    G_lipid = nx.read_edgelist(FNAME_NETWORK)
    pdgm_lipid = pdgm.readwrite.read_pdiagram(FNAME_PDGM)
    return G_lipid, pdgm_lipid


def test_lipid_network(lipid_data):
    G_lipid, pdgm_lipid = lipid_data
    pdgm_obs = rty.pdiagram_from_network(G_lipid)

    for pt1, pt2 in zip(pdgm_obs, pdgm_lipid, strict=True):
        assert [pt1.birth, pt1.death] == pytest.approx([pt2.birth, pt2.death], abs=1e-5)


def test_lipid_network_with_integer_labels(lipid_data):
    G_lipid, pdgm_lipid = lipid_data
    G = G_lipid.copy()
    G = nx.convert_node_labels_to_integers(G)
    pdgm_obs = rty.pdiagram_from_network(G)

    for pt1, pt2 in zip(pdgm_obs, pdgm_lipid, strict=True):
        assert [pt1.birth, pt1.death] == pytest.approx([pt2.birth, pt2.death], abs=1e-5)


def test_lipid_network_without_self_loops(lipid_data):
    G_lipid, pdgm_lipid = lipid_data
    G = G_lipid.copy()
    G.remove_edges_from(nx.selfloop_edges(G_lipid))
    pdgm_obs = rty.pdiagram_from_network(G)

    for pt1, pt2 in zip(pdgm_obs, pdgm_lipid, strict=True):
        assert [pt1.birth, pt1.death] == pytest.approx([pt2.birth, pt2.death], abs=1e-5)


def test_disconnection_error():
    G = nx.erdos_renyi_graph(10, 0.5)
    G.add_node(11)
    with pytest.raises(Exception):
        # For some reason I get an error here when I use DisconnectedGraphError
        cents.net_flow(G)
