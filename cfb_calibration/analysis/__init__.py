"""Calibration metrics and segmentation analysis."""

from .calibration import (
    CalibrationSummary,
    ReliabilityData,
    brier_score,
    calibration_by_segment,
    calibration_summary,
    calibration_summary_table,
    expected_calibration_error,
    log_loss,
    maximum_calibration_error,
    reliability_diagram_data,
)
from .segmentation import (
    calibration_grid,
    clock_score_diff_grid,
    overconfidence_profile,
)

__all__ = [
    "ReliabilityData",
    "CalibrationSummary",
    "reliability_diagram_data",
    "brier_score",
    "log_loss",
    "expected_calibration_error",
    "maximum_calibration_error",
    "calibration_summary",
    "calibration_by_segment",
    "calibration_summary_table",
    "calibration_grid",
    "clock_score_diff_grid",
    "overconfidence_profile",
]
