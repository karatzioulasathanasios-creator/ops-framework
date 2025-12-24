"""
OPS Framework - Operational Projective Stability
================================================

A framework for detecting projective common-mode structure in multi-channel time series.

Author: Athanasios Karatzioulas
Repository: https://github.com/karatzioulasathanasios-creator/ops-framework
Paper: "AT2020mot: exceptional polarization revealed by a cross-domain projective structure analysis"
"""

__version__ = "1.0.0"
__author__ = "Athanasios Karatzioulas"

from .core import (
    projective_structure_detection,
    calculate_projectivity_index,
    rolling_window_analysis,
    null_calibration
)
from .domains import (
    earth_rotation_analysis,
    pta_analysis,
    tde_polarization_analysis
)