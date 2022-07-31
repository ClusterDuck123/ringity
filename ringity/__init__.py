name = "ringity"
__author__ = "Markus K. Youssef"
__version__ = "0.2a8"

import ringity.classes as classes

from ringity.core          import ring_score, diagram, pdiagram_from_point_cloud, ring_score_from_point_cloud
from ringity.centralities  import *
from ringity.plots         import *
#from ringity.distribution_functions import *
from ringity.graphlet_coefficients import *
from ringity.generators.network_models import network_model
from ringity.gtda import vietoris_rips_from_point_cloud
from ringity.single_cell import get_cell_cycle_genes

from ringity.readwrite.diagram import *
from ringity.generators.diagram import *
from ringity.ring_scores import ring_score_from_sequence

import sys

# Check for appropriate Python version
if sys.version_info[:2] < (3, 6):
    m = "Python 3.6 or later is required for ringity (%d.%d detected)."
    raise ImportError(m % sys.version_info[:2])
