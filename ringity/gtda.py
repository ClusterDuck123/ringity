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


def ring_score(X, flavour = 'geometric', base = None):
    dgm = vietoris_rips_from_point_cloud(X)
    if flavour == 'geometric':
        return dgm.ring_score
    elif flavour == 'linear':
        return np.log2(dgm.ring_score + 1)
    elif flavour == 'entropy':
        if base is None:
            base = np.exp(1)
        return 1 - np.log(base)*ss.entropy(dgm.sequence, base = base) / np.log(len(dgm.trimmed()))
    else:
        raise Error(f"Flavour {flavour} unknown.")
        