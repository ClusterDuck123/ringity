import numpy as np

from ringity.tda.pdiagram.pdiagram import PDiagram

import importlib.util

ripser_spec = importlib.util.find_spec("ripser")

if ripser_spec is not None:
    from ripser import ripser
    PHOMOLOGY = "ripser"
else:
    gtda_spec = importlib.util.find_spec("gtda")
    if gtda_spec is None:
        raise ImportError(
            "Neither ripser nor gtda could be imported. Please install one of them."
        )
    else:
        from gtda.homology import VietorisRipsPersistence
        PHOMOLOGY = "gtda"

# =============================================================================
#  ----------------------- VIETORIS-RIPS PERSISTENCE -------------------------
# =============================================================================


def vietoris_rips_from_point_cloud(
    X,
    metric="euclidean",
    metric_params={},
    homology_dimensions=(0, 1),
    collapse_edges=False,
    coeff=2,
    max_edge_length=np.inf,
    infinity_values=None,
    reduced_homology=True,
    n_jobs=None,
):
    VR = VietorisRipsPersistence(
        metric=metric,
        metric_params=metric_params,
        homology_dimensions=homology_dimensions,
        collapse_edges=collapse_edges,
        coeff=coeff,
        max_edge_length=max_edge_length,
        infinity_values=infinity_values,
        reduced_homology=reduced_homology,
        n_jobs=n_jobs,
    )
    pdgm = VR.fit_transform([X])[0]
    return PDiagram.from_gtda(pdgm)


def pdiagram_from_distance_matrix(
    D, dim=1, **kwargs
):
    """Constructs a PDiagram object from a distance matrix."""
    if PHOMOLOGY == "ripser":
        pdgm_rips = ripser(D, maxdim=dim, distance_matrix=True)["dgms"][dim]
        pdgm = PDiagram(pdgm_rips, dim=dim, diameter=D.max())
    elif PHOMOLOGY == "gtda":
        VR = VietorisRipsPersistence(
            metric="precomputed", homology_dimensions=tuple(range(dim + 1))
        )
        pdgm_gtda = VR.fit_transform([D])[0]
        pdgm = PDiagram.from_gtda(pdgm_gtda, dim=dim, diameter=D.max())
    else:
        raise Exception(f"Unknown phomology backend `{PHOMOLOGY}`.")
    return pdgm
