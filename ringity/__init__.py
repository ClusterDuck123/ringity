name = "ringity"
__author__ = "Markus K. Youssef"
__version__ = "1.0a0"

import ringity.classes as classes

from ringity.core.ringscore_functions import *
from ringity.core.singlecell import *
from ringity.networkmeasures.centralities import *
from ringity.plotting.plot_functions import *
from ringity.networkmeasures.graphlet_coefficients import *
from ringity.generators.network_models import network_model
from ringity.core.phomology import vietoris_rips_from_point_cloud
from ringity.core.genesets import get_cell_cycle_genes
from ringity.core.statistics import *

from ringity.readwrite.diagram import *
from ringity.generators.pdiagrams import *
from ringity.core.ringscore_flavours import ring_score_from_sequence

import sys

# Check for appropriate Python version
if sys.version_info[:2] < (3, 6):
    m = "Python 3.6 or later is required for ringity (%d.%d detected)."
    raise ImportError(m % sys.version_info[:2])
