"""
Calibration metrics for probabilistic win predictions.

Core functions:
  - reliability_diagram_data: bins predictions and computes mean pred vs actual freq
  - brier_score: mean squared error of probability predictions
  - log_loss: cross-entropy loss
  - expected_calibration_error (ECE): weighted average |mean_pred - actual_freq|
  - maximum_calibration_error (MCE): worst-case bin calibration error
  - calibration_summary: run all metrics at once
  - calibration_by_segment: apply metrics to subsets of a DataFrame
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_N_BINS = 10


@dataclass
class ReliabilityData:
    """Output of reliability_diagram_data."""
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    mean_predicted: np.ndarray   # mean predicted probability per bin
    actual_fraction: np.ndarray  # fraction of positive outcomes per bin
    counts: np.ndarray           # number of predictions per bin
    # Derived
    calibration_gap: np.ndarray  # mean_predicted - actual_fraction (+ = overconfident)

    @property
    def n_bins(self) -> int:
        return len(self.bin_centers)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "bin_center": self.bin_centers,
            "mean_predicted": self.mean_predicted,
            "actual_fraction": self.actual_fraction,
            "count": self.counts,
            "calibration_gap": self.calibration_gap,
        })


@dataclass
class CalibrationSummary:
    """All calibration metrics for a set of predictions."""
    n: int
    brier_score: float
    log_loss: float
    ece: float           # Expected Calibration Error
    mce: float           # Maximum Calibration Error
    overconfidence: float  # mean(predicted - actual) when predicted > actual
    underconfidence: float # mean(actual - predicted) when actual > predicted
    reliability: ReliabilityData
    segment_label: str = ""
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"CalibrationSummary(n={self.n}, brier={self.brier_score:.4f}, "
            f"ece={self.ece:.4f}, mce={self.mce:.4f}, log_loss={self.log_loss:.4f})"
        )


def reliability_diagram_data(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = _DEFAULT_N_BINS,
) -> ReliabilityData:
    """
    Compute reliability diagram data.

    Args:
        y_prob: Array of predicted probabilities in [0, 1].
        y_true: Array of binary outcomes (0 or 1).
        n_bins: Number of equal-width probability bins.

    Returns:
        ReliabilityData with per-bin statistics.
    """
    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    mask = ~(np.isnan(y_prob) | np.isnan(y_true))
    y_prob = y_prob[mask]
    y_true = y_true[mask]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    mean_predicted = np.full(n_bins, np.nan)
    actual_fraction = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include upper edge only for the last bin
        if i < n_bins - 1:
            mask_bin = (y_prob >= lo) & (y_prob < hi)
        else:
            mask_bin = (y_prob >= lo) & (y_prob <= hi)

        n = mask_bin.sum()
        counts[i] = n
        if n > 0:
            mean_predicted[i] = y_prob[mask_bin].mean()
            actual_fraction[i] = y_true[mask_bin].mean()

    calibration_gap = mean_predicted - actual_fraction

    return ReliabilityData(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        mean_predicted=mean_predicted,
        actual_fraction=actual_fraction,
        counts=counts,
        calibration_gap=calibration_gap,
    )


def brier_score(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error of probabilistic predictions. Lower = better (0 = perfect)."""
    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    mask = ~(np.isnan(y_prob) | np.isnan(y_true))
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((y_prob[mask] - y_true[mask]) ** 2))


def log_loss(y_prob: np.ndarray, y_true: np.ndarray, eps: float = 1e-7) -> float:
    """Cross-entropy loss. Lower = better."""
    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    mask = ~(np.isnan(y_prob) | np.isnan(y_true))
    if mask.sum() == 0:
        return float("nan")
    p = np.clip(y_prob[mask], eps, 1 - eps)
    y = y_true[mask]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(
    rel: ReliabilityData,
    weighted: bool = True,
) -> float:
    """
    Expected Calibration Error.

    ECE = Σ (n_bin / n_total) * |mean_pred_bin - actual_freq_bin|

    If weighted=False, all bins with data are weighted equally.
    """
    valid = ~np.isnan(rel.mean_predicted) & ~np.isnan(rel.actual_fraction)
    if valid.sum() == 0:
        return float("nan")

    gaps = np.abs(rel.calibration_gap[valid])
    if weighted:
        total = rel.counts[valid].sum()
        if total == 0:
            return float("nan")
        weights = rel.counts[valid] / total
    else:
        weights = np.ones(valid.sum()) / valid.sum()

    return float(np.sum(weights * gaps))


def maximum_calibration_error(rel: ReliabilityData) -> float:
    """Maximum absolute calibration error across bins with data."""
    valid = ~np.isnan(rel.calibration_gap)
    if valid.sum() == 0:
        return float("nan")
    return float(np.nanmax(np.abs(rel.calibration_gap)))


def calibration_summary(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = _DEFAULT_N_BINS,
    label: str = "",
) -> CalibrationSummary:
    """Compute all calibration metrics for a set of predictions."""
    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    mask = ~(np.isnan(y_prob) | np.isnan(y_true))
    n = int(mask.sum())

    if n == 0:
        logger.warning("No valid predictions for segment '%s'", label)
        empty_rel = reliability_diagram_data(np.array([]), np.array([]), n_bins)
        return CalibrationSummary(
            n=0, brier_score=float("nan"), log_loss=float("nan"),
            ece=float("nan"), mce=float("nan"),
            overconfidence=float("nan"), underconfidence=float("nan"),
            reliability=empty_rel, segment_label=label,
        )

    rel = reliability_diagram_data(y_prob[mask], y_true[mask], n_bins)
    ece = expected_calibration_error(rel)
    mce = maximum_calibration_error(rel)
    bs = brier_score(y_prob[mask], y_true[mask])
    ll = log_loss(y_prob[mask], y_true[mask])

    gap = rel.calibration_gap
    # Overconfidence: bins where model predicts higher than actual
    over_mask = (~np.isnan(gap)) & (gap > 0)
    under_mask = (~np.isnan(gap)) & (gap < 0)
    overconf = float(np.mean(gap[over_mask])) if over_mask.any() else 0.0
    underconf = float(np.mean(-gap[under_mask])) if under_mask.any() else 0.0

    return CalibrationSummary(
        n=n,
        brier_score=bs,
        log_loss=ll,
        ece=ece,
        mce=mce,
        overconfidence=overconf,
        underconfidence=underconf,
        reliability=rel,
        segment_label=label,
    )


def calibration_by_segment(
    df: pd.DataFrame,
    segment_col: str,
    prob_col: str = "home_win_prob",
    outcome_col: str = "actual_home_win",
    n_bins: int = _DEFAULT_N_BINS,
    min_samples: int = 30,
) -> dict[str, CalibrationSummary]:
    """
    Compute calibration metrics for each unique value of segment_col.

    Args:
        df: DataFrame of plays.
        segment_col: Column to group by (e.g. 'game_clock_bin').
        prob_col: Column with predicted win probability.
        outcome_col: Column with binary outcome.
        n_bins: Number of bins for reliability diagram.
        min_samples: Minimum plays in a segment to compute metrics.

    Returns:
        Dict mapping segment value → CalibrationSummary.
    """
    results = {}
    for seg_val, group in df.groupby(segment_col, observed=True):
        label = str(seg_val)
        if len(group) < min_samples:
            logger.debug("Segment '%s' has only %d samples, skipping.", label, len(group))
            continue
        results[label] = calibration_summary(
            group[prob_col].values,
            group[outcome_col].values,
            n_bins=n_bins,
            label=label,
        )
    return results


def calibration_summary_table(summaries: dict[str, CalibrationSummary]) -> pd.DataFrame:
    """Convert a dict of CalibrationSummary objects to a summary DataFrame."""
    rows = []
    for label, cs in summaries.items():
        rows.append({
            "segment": label,
            "n": cs.n,
            "brier_score": cs.brier_score,
            "log_loss": cs.log_loss,
            "ece": cs.ece,
            "mce": cs.mce,
            #"overconfidence": cs.overconfidence,
            #"underconfidence": cs.underconfidence,
        })
    return pd.DataFrame(rows)
