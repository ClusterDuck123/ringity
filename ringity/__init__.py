name = "ringity"
__author__ = "Markus K. Youssef"
__version__ = "0.2a4"

from ringity.core          import *
from ringity.centralities  import *
from ringity.plots         import *
from ringity.distribution_functions import *
from ringity.graphlet_coefficients import *
from ringity.classes.network_model import network_model, interaction_strength_to_density, density_to_interaction_strength
from ringity.gtda import vietoris_rips_from_point_cloud
from ringity.single_cell import get_cell_cycle_genes

from ringity.readwrite.diagram import *
from ringity.generators.diagram import *

import sys

# Check for proper Python version
if sys.version_info[:2] < (3, 6):
    m = "Python 3.6 or later is required for ringity (%d.%d detected)."
    raise ImportError(m % sys.version_info[:2])
