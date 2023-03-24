from warnings import warn

from ..controller.cycle import Cycle
from ..controller.plotting import (
    cycle_default_score,
    cycle_specified_score,
    plot_cycle_score,
    plot_results_panel_2d,
    plot_results_panel_3d,
)

warn(
    "The `autora.cycle` module is deprecated. "
    "Use the new `autora.controller` module",
    DeprecationWarning,
    stacklevel=2,
)
