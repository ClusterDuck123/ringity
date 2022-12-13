from test_classes import *
from test_methods import *
from test_main_score import *
from _legacy_test_conversions import *
from test_centralities import *
from test_network_model import *
from test_pdiagram_functions import *
from test_ring_score_flavours import *
from test_ring_score_functions import *

import warnings
warnings.filterwarnings("ignore", category = DeprecationWarning)

if __name__ == '__main__':
    unittest.main()
