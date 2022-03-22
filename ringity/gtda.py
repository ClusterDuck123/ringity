import numpy as np
import scipy.stats as ss

from ringity.classes.diagram import PersistenceDiagram
from gtda.homology import VietorisRipsPersistence

def vietoris_rips_from_point_cloud(X,
                          metric='euclidean',
                          metric_params={},
                          homology_dimensions=(0, 1),
                          collapse_edges=False,
                          coeff=2,
                          max_edge_length=np.inf,
                          infinity_values=None,
                          reduced_homology=True,
                          n_jobs=None):
    VR = VietorisRipsPersistence(metric=metric,
                                 metric_params=metric_params,
                                 homology_dimensions=homology_dimensions,
                                 collapse_edges=collapse_edges,
                                 coeff=coeff,
                                 max_edge_length=max_edge_length,
                                 infinity_values=infinity_values,
                                 reduced_homology=reduced_homology,
                                 n_jobs=n_jobs)
    dgm = VR.fit_transform([X])[0]
    return PersistenceDiagram.from_gtda(dgm)
