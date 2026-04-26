"""Tests for cfb_calibration.processing.features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cfb_calibration.processing.features import (
    _conference_tier,
    _down_distance_situation,
    _game_clock_bin,
    _score_differential_bin,
    _spread_to_win_prob,
    add_features,
)


# ---------------------------------------------------------------------------
# Score differential bins
# ---------------------------------------------------------------------------

class TestScoreDifferentialBin:
    def test_blowout_home(self):
        s = pd.Series([35])
        result = _score_differential_bin(s)
        assert result.iloc[0] == "home_blowout"

    def test_blowout_away(self):
        s = pd.Series([-35])
        result = _score_differential_bin(s)
        assert result.iloc[0] == "away_blowout"

    def test_close_home(self):
        s = pd.Series([3])
        result = _score_differential_bin(s)
        assert result.iloc[0] == "home_close"

    def test_close_away(self):
        s = pd.Series([-3])
        result = _score_differential_bin(s)
        assert result.iloc[0] == "away_close"

    def test_tied(self):
        # 0 differential → boundary; pandas cut with right=True puts 0 in the right bin
        s = pd.Series([0])
        result = _score_differential_bin(s)
        assert result.iloc[0] in ("away_close", "home_close")

    def test_series_length_preserved(self):
        s = pd.Series(range(-40, 41))
        result = _score_differential_bin(s)
        assert len(result) == len(s)


# ---------------------------------------------------------------------------
# Game clock bins
# ---------------------------------------------------------------------------

class TestGameClockBin:
    def _make_df(self, pcts, ot_flags=None):
        if ot_flags is None:
            ot_flags = [False] * len(pcts)
        return pd.DataFrame({"game_clock_pct": pcts, "is_overtime": ot_flags})

    def test_q1(self):
        df = self._make_df([0.1])
        result = _game_clock_bin(df)
        assert result.iloc[0] == "Q1"

    def test_q2(self):
        df = self._make_df([0.4])
        result = _game_clock_bin(df)
        assert result.iloc[0] == "Q2"

    def test_q4(self):
        df = self._make_df([0.9])
        result = _game_clock_bin(df)
        assert result.iloc[0] == "Q4"

    def test_overtime(self):
        df = self._make_df([1.0], ot_flags=[True])
        result = _game_clock_bin(df)
        assert result.iloc[0] == "overtime"

    def test_overtime_overrides_pct(self):
        # Even if game_clock_pct=0.8, if is_overtime=True it should be 'overtime'
        df = self._make_df([0.8], ot_flags=[True])
        result = _game_clock_bin(df)
        assert result.iloc[0] == "overtime"


# ---------------------------------------------------------------------------
# Down & distance situation
# ---------------------------------------------------------------------------

class TestDownDistanceSituation:
    def _make_df(self, downs, distances, yardlines):
        return pd.DataFrame({
            "down": downs,
            "distance": distances,
            "yards_to_endzone": yardlines,
        })

    def test_goalline(self):
        df = self._make_df([1], [1], [3])
        result = _down_distance_situation(df)
        assert result.iloc[0] == "goalline"

    def test_short_yardage(self):
        df = self._make_df([2], [1], [30])
        result = _down_distance_situation(df)
        assert result.iloc[0] == "short_yardage"

    def test_third_and_long(self):
        df = self._make_df([3], [8], [45])
        result = _down_distance_situation(df)
        assert result.iloc[0] == "third_fourth_and_long"

    def test_fourth_and_long(self):
        df = self._make_df([4], [10], [50])
        result = _down_distance_situation(df)
        assert result.iloc[0] == "third_fourth_and_long"

    def test_standard(self):
        df = self._make_df([1], [10], [40])
        result = _down_distance_situation(df)
        assert result.iloc[0] == "standard"


# ---------------------------------------------------------------------------
# Conference tier
# ---------------------------------------------------------------------------

class TestConferenceTier:
    def test_sec_is_p5(self):
        s = pd.Series(["9"])  # SEC conference ID
        result = _conference_tier(s)
        assert result.iloc[0] == "P5"

    def test_big_ten_is_p5(self):
        s = pd.Series(["5"])  # Big Ten
        result = _conference_tier(s)
        assert result.iloc[0] == "P5"

    def test_unknown_conference(self):
        s = pd.Series(["999"])
        result = _conference_tier(s)
        assert result.iloc[0] == "G5_or_FCS"

    def test_none_is_unknown(self):
        s = pd.Series([None])
        result = _conference_tier(s)
        assert result.iloc[0] == "unknown"

    def test_nan_is_unknown(self):
        s = pd.Series([float("nan")])
        result = _conference_tier(s)
        assert result.iloc[0] == "unknown"


# ---------------------------------------------------------------------------
# Spread → win probability
# ---------------------------------------------------------------------------

class TestSpreadToWinProb:
    def test_pick_em_gives_50_pct(self):
        s = pd.Series([0.0])
        result = _spread_to_win_prob(s)
        assert abs(result.iloc[0] - 0.5) < 0.01

    def test_heavy_favorite_high_prob(self):
        # Home favored by 21 → spread = -21
        s = pd.Series([-21.0])
        result = _spread_to_win_prob(s)
        assert result.iloc[0] > 0.85

    def test_heavy_underdog_low_prob(self):
        # Home underdog by 21 → spread = +21
        s = pd.Series([21.0])
        result = _spread_to_win_prob(s)
        assert result.iloc[0] < 0.15

    def test_none_input(self):
        result = _spread_to_win_prob(None)
        assert result is None

    def test_nan_propagated(self):
        s = pd.Series([float("nan")])
        result = _spread_to_win_prob(s)
        assert result.iloc[0] is None or (isinstance(result.iloc[0], float) and np.isnan(result.iloc[0]))


# ---------------------------------------------------------------------------
# Integration: add_features
# ---------------------------------------------------------------------------

class TestAddFeatures:
    def test_adds_all_feature_columns(self):
        df = pd.DataFrame({
            "score_differential": [0, 14, -7, 28],
            "game_clock_pct": [0.1, 0.4, 0.75, 1.0],
            "is_overtime": [False, False, False, False],
            "down": [1, 2, 3, 4],
            "distance": [10, 5, 8, 1],
            "yards_to_endzone": [60, 45, 30, 3],
            "home_conference": ["9", "5", "1", "999"],
            "away_conference": ["9", "1", "999", "5"],
            "group": [80, 80, 80, 80],
        })
        result = add_features(df)
        for col in ["score_differential_bin", "game_clock_bin",
                    "down_distance_situation", "conference_tier_home",
                    "conference_tier_away"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_preserved(self):
        df = pd.DataFrame({
            "score_differential": range(20),
            "game_clock_pct": np.linspace(0, 1, 20),
            "is_overtime": [False] * 20,
            "down": [1] * 20,
            "distance": [10] * 20,
            "yards_to_endzone": [50] * 20,
            "home_conference": ["9"] * 20,
            "away_conference": ["5"] * 20,
            "group": [80] * 20,
        })
        result = add_features(df)
        assert len(result) == len(df)
