name = "ringity"
__author__ = "Markus K. Youssef"
__version__ = "0.3a0"

# FUNCTIONS
from ringity.core import (
                    pdiagram, 
                    pdiagram_from_network,
                    pdiagram_from_point_cloud,
                    pdiagram_from_distance_matrix,
                    
                    ring_score,
                    ring_score_from_network,
                    ring_score_from_point_cloud,
                    ring_score_from_distance_matrix,

                    ring_score_from_sequence,
                    ring_score_from_pdiagram,

                    pwdistance,
                    pwdistance_from_network,
                    pwdistance_from_point_cloud,
                    pwdistance_from_adjacency_matrix
                    )

# MODULES
from ringity.core import (statistics)

from ringity.generators import (
                    random_pdgm,
                    network_model,
                    )
import ringity.generators.geometric_networks as geometric_networks

from ringity.plotting import (plot, plot_nx, plot_dgm)

from ringity.readwrite.pdiagrams import (write_pdiagram, read_pdiagram)

from ringity.singlecell import singlecell
from ringity.classes.pdiagram import PDiagram

import sys
# Check for appropriate Python version
major, minor = sys.version_info[:2]
if (major, minor) < (3, 6):
    msg = f'Python 3.6 or later is required for ringity ' \
          f'({major}.{minor} detected).'
    raise ImportError(msg)
