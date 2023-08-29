name = "ringity"
__author__ = "M. K. Youssef, et al."
__version__ = "0.3a4"

# FUNCTIONS
from ringity.ringscore.metric2ringscore import ring_score_from_sequence
from ringity.ringscore import (
                    pdiagram, 
                    pdiagram_from_network,
                    pdiagram_from_point_cloud,
                    pdiagram_from_distance_matrix,
                    
                    ring_score,
                    ring_score_from_network,
                    ring_score_from_point_cloud,
                    ring_score_from_distance_matrix,

                    ring_score_from_pdiagram,

                    pwdistance,
                    pwdistance_from_network,
                    pwdistance_from_point_cloud,
                    pwdistance_from_adjacency_matrix
                    )

# MODULES
from ringity.ringscore import (statistics)

from ringity.networks.networkmodel import network_model
import ringity.networks.geometric_networks as geometric_networks

from ringity.utils.plotting import (
                        plot,
                        plot_X,
                        plot_nx,
                        plot_dgm,
                        plot_seq,
                        plot_degree_distribution)

from ringity.tda.pdiagram.readwrite import (write_pdiagram, read_pdiagram)

from ringity.tda.pdiagram.pdiagram import PDiagram

import sys
# Check for appropriate Python version
major, minor = sys.version_info[:2]
if (major, minor) < (3, 6):
    msg = f'Python 3.6 or later is required for ringity ' \
          f'({major}.{minor} detected).'
    raise ImportError(msg)
