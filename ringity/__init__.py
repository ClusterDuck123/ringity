name = "ringity"
__author__ = "Markus Kirolos Youssef"
__version__ = "0.1a0"

from ringity.methods       import *
from ringity.core          import *
from ringity.centralities  import *
from ringity.plots         import *
from ringity.network_model import *
from ringity.distribution_functions import *

import sys
import ringity.test_venv_compability

# Check for proper Python version
if sys.version_info[:2] < (3, 6):
    m = "Python 3.6 or later is required for ringity (%d.%d detected)."
    raise ImportError(m % sys.version_info[:2])
