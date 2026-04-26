"""
Feature engineering for the plays DataFrame.

Adds derived columns used for segmented calibration analysis:
  - score_differential_bin
  - game_clock_bin
  - down_distance_situation
  - conference_tier
  - spread_category
  - is_close_game (score_diff within 8 at any point in Q4/OT)
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Conference tier mapping
# ---------------------------------------------------------------------------

# Power Four (formerly Power Five) conferences after 2024 realignment
_P4_CONFERENCE_IDS = {
    "1",   # ACC
    "4",   # Big 12
    "5",   # Big Ten
    "8",   # Pac-12
    "9",   # SEC
}

# Treat these as P5/Power for historical purposes
_POWER_CONFERENCE_NAMES = {
    "acc", "big 12", "big ten", "pac-12", "pac-10", "sec",
    "atlantic coast conference", "big twelve",
}

_FBS_INDEPENDENT_NAMES = {"ind", "independent", "fbs ind", "fbs independent"}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered feature columns to a plays DataFrame in-place.

    Expects columns: score_differential, game_clock_pct, period, down, distance,
                     yards_to_endzone, home_conference, away_conference, group.
    """
    df = df.copy()
    df["score_differential_bin"] = _score_differential_bin(df["score_differential"])
    df["game_clock_bin"] = _game_clock_bin(df)
    df["down_distance_situation"] = _down_distance_situation(df)
    df["conference_tier_home"] = _conference_tier(df["home_conference"])
    df["conference_tier_away"] = _conference_tier(df["away_conference"])
    df["game_tier"] = _game_tier(df)
    return df


def add_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add spread-based features. Requires pre_game_spread column."""
    df = df.copy()
    df["spread_category"] = _spread_category(df.get("pre_game_spread"))
    df["implied_home_win_prob_vegas"] = _spread_to_win_prob(df.get("pre_game_spread"))
    return df


# ---------------------------------------------------------------------------
# Score differential bins
# ---------------------------------------------------------------------------

_SCORE_BINS = [-100, -28, -21, -14, -7, 0, 7, 14, 21, 28, 100]
_SCORE_LABELS = [
    "away_blowout",     # < -28
    "away_large",       # -28 to -21
    "away_medium",      # -21 to -14
    "away_small",       # -14 to -7
    "away_close",       # -7 to 0
    "home_close",       # 0 to 7
    "home_small",       # 7 to 14
    "home_medium",      # 14 to 21
    "home_large",       # 21 to 28
    "home_blowout",     # > 28
]


def _score_differential_bin(series: pd.Series) -> pd.Series:
    return pd.cut(series, bins=_SCORE_BINS, labels=_SCORE_LABELS, right=True)


# ---------------------------------------------------------------------------
# Game clock bins
# ---------------------------------------------------------------------------

def _game_clock_bin(df: pd.DataFrame) -> pd.Series:
    """
    Categorize each play into a game-time bucket.
    OT plays are labeled 'overtime' regardless of clock_pct.
    """
    bins = pd.cut(
        df["game_clock_pct"],
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=["Q1", "Q2", "Q3", "Q4"],
        include_lowest=True,
        right=True,
    ).astype(object)

    is_ot = df.get("is_overtime", pd.Series(False, index=df.index))
    bins[is_ot.fillna(False)] = "overtime"
    return bins


# ---------------------------------------------------------------------------
# Down & distance situation
# ---------------------------------------------------------------------------

def _down_distance_situation(df: pd.DataFrame) -> pd.Series:
    """
    Classify play situation into a handful of meaningful categories.

    Categories:
      goalline         — inside the 5-yard line
      short_yardage    — 1–2 yards to go
      third_or_fourth_and_long — 3rd/4th & 5+
      standard         — everything else
    """
    down = df.get("down", pd.Series(dtype="Int64"))
    distance = df.get("distance", pd.Series(dtype="Int64"))
    yte = df.get("yards_to_endzone", pd.Series(dtype="Int64"))

    conditions = [
        (yte.notna()) & (yte <= 5),
        (distance.notna()) & (distance <= 2),
        (down.isin([3, 4])) & (distance.notna()) & (distance >= 5),
    ]
    choices = ["goalline", "short_yardage", "third_fourth_and_long"]
    result = pd.Series("standard", index=df.index)
    for cond, label in zip(reversed(conditions), reversed(choices)):
        result[cond] = label
    return result


# ---------------------------------------------------------------------------
# Conference tier
# ---------------------------------------------------------------------------

def _conference_tier(conf_series: pd.Series | None) -> pd.Series:
    """
    Map conference IDs or names to tier labels: 'P5', 'G5', 'FCS', 'unknown'.
    """
    if conf_series is None:
        return pd.Series("unknown")

    def _classify(val) -> str:
        if pd.isna(val):
            return "unknown"
        s = str(val).strip().lower()
        if s in _P4_CONFERENCE_IDS:
            return "P5"
        if any(p in s for p in _POWER_CONFERENCE_NAMES):
            return "P5"
        if s in _FBS_INDEPENDENT_NAMES:
            return "G5"  # treat FBS independents as G5-level
        # If numeric ID not in P4, it's G5 (if FBS) or FCS
        return "G5_or_FCS"

    return conf_series.map(_classify)


def _game_tier(df: pd.DataFrame) -> pd.Series:
    """
    Overall game tier based on the group column (80=FBS, 81=FCS).
    For FBS vs FBS games, further classify by conference.
    """
    group = df.get("group", pd.Series(dtype=object))

    def _tier(row):
        grp = row.get("group")
        if grp == 81:
            return "FCS"
        # FBS game
        home_tier = row.get("conference_tier_home", "unknown")
        away_tier = row.get("conference_tier_away", "unknown")
        tiers = {home_tier, away_tier}
        if "P5" in tiers and "P5" in tiers and len(tiers) == 1:
            return "P5_vs_P5"
        if "P5" in tiers:
            return "P5_vs_G5_or_FCS"
        return "G5_vs_G5"

    return df.apply(_tier, axis=1)


# ---------------------------------------------------------------------------
# Spread-based features
# ---------------------------------------------------------------------------

_SPREAD_BINS = [float("-inf"), -21, -14, -7, 7, 14, 21, float("inf")]
_SPREAD_LABELS = [
    "home_heavy_fav",   # spread < -21 (home favored by 21+)
    "home_large_fav",   # -21 to -14
    "home_mod_fav",     # -14 to -7
    "tossup",           # -7 to +7
    "away_mod_fav",     # +7 to +14
    "away_large_fav",   # +14 to +21
    "away_heavy_fav",   # > +21
]


def _spread_category(spread_series: pd.Series | None) -> pd.Series:
    if spread_series is None:
        return pd.Series("unknown")
    return pd.cut(spread_series, bins=_SPREAD_BINS, labels=_SPREAD_LABELS, right=True)


def _spread_to_win_prob(spread_series: pd.Series | None) -> pd.Series | None:
    """
    Convert a Vegas point spread to an implied win probability using the
    standard normal distribution approximation.

    P(home win) ≈ Φ(−spread / σ), where σ ≈ 13.45 points (historical std).
    """
    if spread_series is None:
        return None
    import numpy as np
    from scipy.stats import norm

    sigma = 13.45
    return spread_series.map(
        lambda s: float(norm.cdf(-s / sigma)) if pd.notna(s) else None
    )
