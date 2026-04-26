"""
Publication-quality calibration visualizations.

All functions return (fig, ax) or (fig, axes) for easy customization.
Default style is suitable for Towards Data Science / Medium blog posts.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from ..analysis.calibration import CalibrationSummary, ReliabilityData

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

_PALETTE = sns.color_palette("colorblind")
_PERFECT_COLOR = "#aaaaaa"
_MAIN_COLOR = _PALETTE[0]
_SECONDARY_COLOR = _PALETTE[1]
_OVERCONF_COLOR = _PALETTE[3]   # red-ish
_UNDERCONF_COLOR = _PALETTE[2]  # green-ish

plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def reliability_diagram(
    rel: ReliabilityData,
    *,
    title: str = "Reliability Diagram",
    subtitle: str = "",
    ece: float | None = None,
    brier: float | None = None,
    n: int | None = None,
    ax: plt.Axes | None = None,
    show_histogram: bool = True,
    color: str = _MAIN_COLOR,
) -> tuple[plt.Figure, plt.Axes | tuple[plt.Axes, plt.Axes]]:
    """
    Plot a reliability (calibration) diagram with optional prediction histogram.

    Args:
        rel: ReliabilityData from reliability_diagram_data().
        title: Main title.
        subtitle: Optional subtitle shown below title.
        ece: Expected calibration error (shown in legend if provided).
        brier: Brier score (shown in legend if provided).
        n: Total sample count (shown in subtitle area).
        ax: Existing axes to draw on. If None, creates a new figure.
        show_histogram: Whether to show the prediction distribution histogram below.
        color: Color for the calibration line.

    Returns:
        (fig, ax) if show_histogram=False, else (fig, (ax_cal, ax_hist)).
    """
    valid = ~(np.isnan(rel.mean_predicted) | np.isnan(rel.actual_fraction))

    if ax is None:
        if show_histogram:
            fig, (ax_cal, ax_hist) = plt.subplots(
                2, 1, figsize=(7, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
            )
        else:
            fig, ax_cal = plt.subplots(figsize=(7, 6))
            ax_hist = None
    else:
        fig = ax.figure
        ax_cal = ax
        ax_hist = None
        show_histogram = False

    # --- Calibration curve ---
    ax_cal.plot([0, 1], [0, 1], "--", color=_PERFECT_COLOR, linewidth=1.5, label="Perfect calibration", zorder=1)

    label_parts = []
    if ece is not None:
        label_parts.append(f"ECE = {ece:.3f}")
    if brier is not None:
        label_parts.append(f"Brier = {brier:.3f}")
    legend_label = "ESPN model  (" + ", ".join(label_parts) + ")" if label_parts else "ESPN model"

    ax_cal.plot(
        rel.mean_predicted[valid],
        rel.actual_fraction[valid],
        "o-",
        color=color,
        linewidth=2,
        markersize=7,
        label=legend_label,
        zorder=3,
    )

    # Shade calibration gap areas
    ax_cal.fill_between(
        rel.mean_predicted[valid],
        rel.actual_fraction[valid],
        rel.mean_predicted[valid],
        where=rel.mean_predicted[valid] > rel.actual_fraction[valid],
        alpha=0.12,
        color=_OVERCONF_COLOR,
        label="Overconfident",
    )
    ax_cal.fill_between(
        rel.mean_predicted[valid],
        rel.actual_fraction[valid],
        rel.mean_predicted[valid],
        where=rel.mean_predicted[valid] < rel.actual_fraction[valid],
        alpha=0.12,
        color=_UNDERCONF_COLOR,
        label="Underconfident",
    )

    ax_cal.set_xlim(-0.02, 1.02)
    ax_cal.set_ylim(-0.02, 1.02)
    ax_cal.set_ylabel("Actual win rate", fontsize=12)
    ax_cal.legend(fontsize=7, loc="upper left", handlelength=1, borderpad=0.4)
    ax_cal.set_aspect("equal")

    full_title = title
    if subtitle:
        full_title = f"{title}\n{subtitle}"
    if n is not None:
        full_title += f"\n(n = {n:,} plays)"
    ax_cal.set_title(full_title, fontsize=13, pad=10)

    # --- Histogram ---
    if show_histogram and ax_hist is not None:
        ax_hist.bar(
            rel.bin_centers,
            rel.counts / max(rel.counts.sum(), 1),
            width=rel.bin_centers[1] - rel.bin_centers[0] if len(rel.bin_centers) > 1 else 0.1,
            color=color,
            alpha=0.6,
            align="center",
        )
        ax_hist.set_xlabel("Predicted win probability", fontsize=12)
        ax_hist.set_ylabel("Fraction\nof plays", fontsize=10)
        ax_hist.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax_hist.set_xlim(-0.02, 1.02)

        plt.tight_layout()
        return fig, (ax_cal, ax_hist)

    ax_cal.set_xlabel("Predicted win probability", fontsize=12)
    plt.tight_layout()
    return fig, ax_cal


# ---------------------------------------------------------------------------
# Multi-segment reliability diagram
# ---------------------------------------------------------------------------

def reliability_diagram_grid(
    summaries: dict[str, CalibrationSummary],
    *,
    cols: int = 3,
    title: str = "Calibration by Segment",
    figsize_per_panel: tuple[float, float] = (4.5, 4.5),
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot a grid of reliability diagrams, one per segment.
    """
    labels = list(summaries.keys())
    n_panels = len(labels)
    rows = (n_panels + cols - 1) // cols

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
    )
    axes_flat = np.array(axes).ravel()

    for i, label in enumerate(labels):
        cs = summaries[label]
        ax = axes_flat[i]
        reliability_diagram(
            cs.reliability,
            title=label,
            ece=cs.ece,
            brier=cs.brier_score,
            n=cs.n,
            ax=ax,
            show_histogram=False,
        )

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=15, y=1.01)
    plt.tight_layout()
    return fig, axes_flat


# ---------------------------------------------------------------------------
# ECE over game clock
# ---------------------------------------------------------------------------

def brier_ece_by_clock(
    clock_summaries: dict[str, CalibrationSummary],
    *,
    title: str = "Calibration vs. Game Clock",
    dual_plot: bool = True,
    max_index: int = None
) -> tuple[plt.Figure, tuple[plt.Axes, ...]]:
    """
    Line plot of Brier score and ECE across game clock buckets.

    dual_plot=True (default): shared figure with twin y-axes.
    dual_plot=False: two vertically stacked subplots, one per metric.

    clock_summaries should be keyed by ordered clock labels (e.g. 'Q1', 'Q2', ...).
    """
    labels = list(clock_summaries.keys())
    if max_index is None:
        max_index = len(labels)
    labels = labels[:max_index]
    ece_vals = [clock_summaries[l].ece for l in labels]
    brier_vals = [clock_summaries[l].brier_score for l in labels]

    x = np.arange(len(labels))
    step = max(1, len(labels) // 6)
    tick_positions = x[::step]
    tick_labels = labels[::step]

    def _apply_xticks(ax: plt.Axes) -> None:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=10, rotation=45, ha="right")

    if dual_plot:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()

        ax1.plot(x, brier_vals, "o-", color=_MAIN_COLOR, linewidth=2, markersize=3, label="Brier Score")
        ax2.plot(x, ece_vals, "s--", color=_SECONDARY_COLOR, linewidth=2, markersize=3, label="ECE")

        _apply_xticks(ax1)
        ax1.set_xlabel("Game clock segment", fontsize=12)
        ax1.set_ylabel("Brier Score", color=_MAIN_COLOR, fontsize=12)
        ax2.set_ylabel("ECE", color=_SECONDARY_COLOR, fontsize=12)
        ax1.set_title(title, fontsize=14)
        plt.tight_layout()
        return fig, (ax1, ax2)
    else:
        fig, (ax_brier, ax_ece) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

        ax_brier.plot(x, brier_vals, "o-", color=_MAIN_COLOR, linewidth=2, markersize=4)
        ax_brier.set_ylabel("Brier Score", fontsize=12)
        ax_brier.set_title(title, fontsize=14)

        ax_ece.plot(x, ece_vals, "s--", color=_SECONDARY_COLOR, linewidth=2, markersize=4)
        ax_ece.set_ylabel("ECE", fontsize=12)
        ax_ece.set_xlabel("Game clock segment", fontsize=12)
        _apply_xticks(ax_ece)

        plt.tight_layout()
        return fig, (ax_brier, ax_ece)


# ---------------------------------------------------------------------------
# Calibration heatmap (2D)
# ---------------------------------------------------------------------------

def calibration_heatmap(
    pivot_df: pd.DataFrame,
    *,
    title: str = "Calibration Error Heatmap",
    xlabel: str = "Score Differential",
    ylabel: str = "Game Clock",
    vmax: float = 0.15,
    figsize: tuple = (14, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a heatmap of ECE values from a pivot table (rows=clock, cols=score_diff).
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot_df,
        ax=ax,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=vmax,
        linewidths=0.4,
        linecolor="white",
        annot=True,
        fmt=".3f",
        annot_kws={"size": 7},
        cbar_kws={"label": "ECE (lower = better calibrated)"},
        mask=pivot_df.isna(),
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Calibration gap distribution
# ---------------------------------------------------------------------------

def calibration_gap_distribution(
    summaries: dict[str, CalibrationSummary],
    *,
    title: str = "Calibration Gap Distribution Across Segments",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Horizontal bar chart showing calibration gap (mean_pred − actual) per segment.
    Positive = overconfident, negative = underconfident.
    """
    labels = list(summaries.keys())
    # Mean calibration gap across all bins (weighted)
    gaps = []
    for cs in summaries.values():
        valid = ~np.isnan(cs.reliability.calibration_gap)
        if valid.any():
            w = cs.reliability.counts[valid] / max(cs.reliability.counts[valid].sum(), 1)
            gaps.append(float(np.sum(w * cs.reliability.calibration_gap[valid])))
        else:
            gaps.append(0.0)

    sorted_pairs = sorted(zip(gaps, labels))
    gaps_s, labels_s = zip(*sorted_pairs)

    colors = [_OVERCONF_COLOR if g > 0 else _UNDERCONF_COLOR for g in gaps_s]

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.4)))
    bars = ax.barh(labels_s, gaps_s, color=colors, edgecolor="white", height=0.7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean calibration gap (predicted − actual)", fontsize=11)
    ax.set_title(title, fontsize=13)

    for bar, gap in zip(bars, gaps_s):
        ax.text(
            gap + (0.001 if gap >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{gap:+.3f}",
            va="center",
            ha="left" if gap >= 0 else "right",
            fontsize=8,
        )

    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Season trend
# ---------------------------------------------------------------------------

def season_trend_plot(
    season_summaries: dict[str | int, CalibrationSummary],
    *,
    title: str = "ESPN Calibration Over Time",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Line plot of ECE and Brier score by season.
    """
    seasons = sorted(season_summaries.keys(), key=int)
    ece_vals = [season_summaries[s].ece for s in seasons]
    brier_vals = [season_summaries[s].brier_score for s in seasons]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(seasons, brier_vals, "o-", color=_MAIN_COLOR, label="Brier Score", linewidth=2, markersize=8)
    ax.plot(seasons, ece_vals, "s--", color=_SECONDARY_COLOR, label="ECE", linewidth=2, markersize=8)

    ax.set_xlabel("Season", fontsize=12)
    ax.set_ylabel("Metric value (lower = better)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks(list(seasons))
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# ESPN vs Vegas comparison
# ---------------------------------------------------------------------------

def espn_vs_vegas_scatter(
    df: pd.DataFrame,
    espn_col: str = "espn_home_prob_pregame",
    vegas_col: str = "implied_home_win_prob_vegas",
    outcome_col: str = "actual_home_win",
    *,
    title: str = "ESPN Pre-game Probability vs. Vegas Implied Probability",
    sample: int = 5000,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot comparing ESPN's pre-game win probability to the Vegas-implied probability.
    Points colored by outcome.
    """
    sub = df[[espn_col, vegas_col, outcome_col]].dropna()
    if len(sub) > sample:
        sub = sub.sample(sample, random_state=42)

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(
        sub[vegas_col],
        sub[espn_col],
        c=sub[outcome_col],
        cmap="RdYlGn",
        alpha=0.4,
        s=15,
        vmin=0,
        vmax=1,
    )
    ax.plot([0, 1], [0, 1], "--", color=_PERFECT_COLOR, linewidth=1.5, label="ESPN = Vegas")
    plt.colorbar(scatter, ax=ax, label="Actual home win (1=yes)")
    ax.set_xlabel("Vegas implied home win probability", fontsize=12)
    ax.set_ylabel("ESPN pre-game home win probability", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# OTE by period
# ---------------------------------------------------------------------------

def ote_by_period(
    ote: pd.DataFrame,
    *,
    title: str = "Over/Under Total Expectations (OTE) by Period",
    xlabel: str = "Period",
    figsize: tuple[float, float] = (10, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Line plot of OTE values by period (or any categorical grouping).

    Args:
        ote: DataFrame with one row ("game_segment") and columns as period labels,
             as returned by get_begin_segment_ote(). Values > 1 = model underestimated,
             < 1 = overestimated.
        title: Plot title.
        xlabel: x-axis label.
        figsize: Figure size.

    Returns:
        (fig, ax)
    """
    x = list(range(len(ote.columns)))
    values = ote.loc["game_segment"].values
    labels = [str(l) for l in ote.columns]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, values, "o-", color=_MAIN_COLOR, linewidth=1.5, markersize=4)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", label="Perfect (OTE = 1.0)")

    ax.xaxis.set_major_locator(mticker.AutoLocator())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda i, _: labels[int(i)] if 0 <= int(i) < len(labels) else ""))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("OTE  (actual wins / predicted wins)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig, ax


def ece_by_period(
    ote: pd.DataFrame,
    *,
    title: str = "ECE by Period",
    xlabel: str = "Period",
    figsize: tuple[float, float] = (10, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Line plot of ECE values by period (or any categorical grouping).

    Args:
        ece: DataFrame with one row ("game_segment") and columns as period labels,
        title: Plot title.
        xlabel: x-axis label.
        figsize: Figure size.

    Returns:
        (fig, ax)
    """
    x = list(range(len(ote.columns)))
    values = ote.loc["game_segment"].values
    labels = [str(l) for l in ote.columns]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, values, "o-", color=_MAIN_COLOR, linewidth=1.5, markersize=4)
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")

    ax.xaxis.set_major_locator(mticker.AutoLocator())
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda i, _: labels[int(i)] if 0 <= int(i) < len(labels) else ""))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("ECE  |predicted - actual|", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    return fig, ax