"""
2D segmentation analysis: calibration error as a function of two variables simultaneously.

Primary use case: game_clock_pct × score_differential heatmap of ECE,
revealing when and in what game states ESPN's model is most/least calibrated.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .calibration import calibration_summary, CalibrationSummary

logger = logging.getLogger(__name__)


def calibration_grid(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    prob_col: str = "home_win_prob",
    outcome_col: str = "actual_home_win",
    n_bins: int = 10,
    min_samples: int = 50,
) -> tuple[pd.DataFrame, dict[tuple, CalibrationSummary]]:
    """
    Compute ECE on a 2D grid of (row_col × col_col) segment pairs.

    Returns:
        pivot_df: DataFrame with row_col values as index, col_col values as columns,
                  cells = ECE (NaN where insufficient data).
        summaries: Dict mapping (row_val, col_val) → CalibrationSummary.
    """
    summaries: dict[tuple, CalibrationSummary] = {}
    records = []

    row_vals = sorted(df[row_col].dropna().unique(), key=str)
    col_vals = sorted(df[col_col].dropna().unique(), key=str)

    for rv in row_vals:
        for cv in col_vals:
            subset = df[(df[row_col] == rv) & (df[col_col] == cv)]
            label = f"{rv}|{cv}"
            if len(subset) < min_samples:
                summaries[(rv, cv)] = None
                records.append({"row": rv, "col": cv, "ece": np.nan, "n": len(subset)})
                continue

            cs = calibration_summary(
                subset[prob_col].values,
                subset[outcome_col].values,
                n_bins=n_bins,
                label=label,
            )
            summaries[(rv, cv)] = cs
            records.append({"row": rv, "col": cv, "ece": cs.ece, "n": cs.n})

    grid_df = pd.DataFrame(records)
    pivot_df = grid_df.pivot(index="row", columns="col", values="ece")
    return pivot_df, summaries


def clock_score_diff_grid(
    df: pd.DataFrame,
    clock_bins: int = 10,
    score_bins: int = 11,
    prob_col: str = "home_win_prob",
    outcome_col: str = "actual_home_win",
    min_samples: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Produce a fine-grained clock × score_differential calibration heatmap.

    Returns:
        ece_pivot: ECE per cell.
        count_pivot: Sample count per cell.
    """
    df = df.copy()

    # Bin game_clock_pct into equal intervals
    df["_clock_bin"] = pd.cut(
        df["game_clock_pct"].clip(0, 1),
        bins=clock_bins,
        labels=[f"{int(i/clock_bins*100)}–{int((i+1)/clock_bins*100)}%" for i in range(clock_bins)],
    )

    # Bin score_differential
    max_diff = 42
    df["_score_bin"] = pd.cut(
        df["score_differential"].clip(-max_diff, max_diff),
        bins=score_bins,
        labels=[
            f"{int(-max_diff + i * 2 * max_diff / score_bins)}–{int(-max_diff + (i+1) * 2 * max_diff / score_bins)}"
            for i in range(score_bins)
        ],
    )

    records = []
    for (cb, sb), subset in df.groupby(["_clock_bin", "_score_bin"], observed=True):
        n = len(subset)
        if n < min_samples:
            ece = np.nan
        else:
            cs = calibration_summary(
                subset[prob_col].values,
                subset[outcome_col].values,
                n_bins=10,
            )
            ece = cs.ece
        records.append({"clock": cb, "score": sb, "ece": ece, "n": n})

    grid_df = pd.DataFrame(records)
    ece_pivot = grid_df.pivot(index="clock", columns="score", values="ece")
    count_pivot = grid_df.pivot(index="clock", columns="score", values="n")
    return ece_pivot, count_pivot


def overconfidence_profile(
    df: pd.DataFrame,
    segment_col: str,
    prob_col: str = "home_win_prob",
    outcome_col: str = "actual_home_win",
) -> pd.DataFrame:
    """
    For each segment, compute the signed calibration gap:
      positive = overconfident (predicted > actual)
      negative = underconfident (predicted < actual)

    Uses mean(predicted - actual) as a simple directional metric.
    """
    rows = []
    for seg_val, group in df.groupby(segment_col, observed=True):
        mask = group[prob_col].notna() & group[outcome_col].notna()
        sub = group[mask]
        if len(sub) < 10:
            continue
        mean_pred = sub[prob_col].mean()
        actual_rate = sub[outcome_col].mean()
        gap = mean_pred - actual_rate
        rows.append({
            "segment": str(seg_val),
            "n": len(sub),
            "mean_predicted": mean_pred,
            "actual_win_rate": actual_rate,
            "calibration_gap": gap,
            "direction": "overconfident" if gap > 0 else "underconfident",
        })
    return pd.DataFrame(rows).sort_values("calibration_gap")
