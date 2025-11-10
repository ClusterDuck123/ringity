from importlib import metadata as _md
from typing import TYPE_CHECKING


# FUNCTIONS
from .ringscore import (
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
    pwdistance_from_adjacency_matrix,
)

# MODULES
from .utils.plotting import (
    plot,
    plot_X,
    plot_nx,
    plot_dgm,
    plot_seq,
    plot_degree_distribution,
)

import sys

# Check for appropriate Python version

major, minor = sys.version_info[:2]
if (major, minor) < (3, 6):
    msg = f"Python 3.6 or later is required for ringity " f"({major}.{minor} detected)."
    raise ImportError(msg)


__all__ = ["set_theme", "ring_score_from_sequence", "network_model"]


def __getattr__(name: str):  # PEP 562
    if name == "set_theme":
        from .utils.plotting import set_theme

        return set_theme
    if name == "ring_score_from_sequence":
        from .ringscore.metric2ringscore import ring_score_from_sequence

        return ring_score_from_sequence
    if name == "network_model":
        from .networkmodel import network_model

        return network_model
    raise AttributeError(name)


def __dir__():
    # make tab-completion in plain Python/IPython list these names
    return sorted(set(globals().keys()) | set(__all__))


if TYPE_CHECKING:
    # visible to type checkers/IDEs, not executed at runtime
    from .utils.plotting import set_theme as set_theme
    from .ringscore.metric2ringscore import (
        ring_score_from_sequence as ring_score_from_sequence,
    )
    from .networkmodel import network_model as network_model

try:
    __version__ = _md.version("ringity")
except _md.PackageNotFoundError:
    try:
        from ._version import version as __version__  # generated during build
    except Exception:
        __version__ = "0.0.0+dev"
