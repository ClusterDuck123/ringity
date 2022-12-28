import numpy as np

from ringity.classes.pdiagram import PDiagram
from gtda.homology import VietorisRipsPersistence # CHANGE FOR FUTURE: OPTIONAL IMPORT

# =============================================================================
#  ----------------------- VIETORIS-RIPS PERSISTENCE -------------------------
# =============================================================================

def vietoris_rips_from_point_cloud(X,
                            metric = 'euclidean',
                            metric_params = {},
                            homology_dimensions = (0, 1),
                            collapse_edges = False,
                            coeff = 2,
                            max_edge_length = np.inf,
                            infinity_values = None,
                            reduced_homology = True,
                            n_jobs = None):
    VR = VietorisRipsPersistence(
                            metric = metric,
                            metric_params = metric_params,
                            homology_dimensions = homology_dimensions,
                            collapse_edges = collapse_edges,
                            coeff = coeff,
                            max_edge_length = max_edge_length,
                            infinity_values = infinity_values,
                            reduced_homology = reduced_homology,
                            n_jobs = n_jobs)
    dgm = VR.fit_transform([X])[0]
    return PDiagram.from_gtda(dgm)

