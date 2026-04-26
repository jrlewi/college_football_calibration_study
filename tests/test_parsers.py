"""Tests for cfb_calibration.processing.parsers."""

from __future__ import annotations

import pytest
from cfb_calibration.processing.parsers import (
    _compute_seconds_elapsed,
    _game_clock_pct,
    _REGULATION_SECONDS,
    parse_game,
)


class TestParseGame:
    def test_returns_game_and_plays(self, sample_espn_summary, sample_game_meta):
        game, plays = parse_game(sample_espn_summary, sample_game_meta)
        assert isinstance(game, dict)
        assert isinstance(plays, list)

    def test_game_id_extracted(self, sample_espn_summary, sample_game_meta):
        game, _ = parse_game(sample_espn_summary, sample_game_meta)
        assert game["game_id"] == "401628333"

    def test_teams_extracted(self, sample_espn_summary, sample_game_meta):
        game, _ = parse_game(sample_espn_summary, sample_game_meta)
        assert "Georgia" in game["home_team"]
        assert "Auburn" in game["away_team"]

    def test_final_scores(self, sample_espn_summary, sample_game_meta):
        game, _ = parse_game(sample_espn_summary, sample_game_meta)
        assert game["home_score_final"] == 30
        assert game["away_score_final"] == 13

    def test_actual_home_win(self, sample_espn_summary, sample_game_meta):
        game, _ = parse_game(sample_espn_summary, sample_game_meta)
        assert game["actual_home_win"] == 1

    def test_play_count(self, sample_espn_summary, sample_game_meta):
        _, plays = parse_game(sample_espn_summary, sample_game_meta)
        # All 4 plays have win probability entries
        assert len(plays) == 4

    def test_win_prob_range(self, sample_espn_summary, sample_game_meta):
        _, plays = parse_game(sample_espn_summary, sample_game_meta)
        for p in plays:
            assert 0.0 <= p["home_win_prob"] <= 1.0

    def test_win_prob_values(self, sample_espn_summary, sample_game_meta):
        _, plays = parse_game(sample_espn_summary, sample_game_meta)
        probs = [p["home_win_prob"] for p in plays]
        assert abs(probs[0] - 0.625) < 1e-6
        assert abs(probs[3] - 0.95) < 1e-6

    def test_play_has_required_fields(self, sample_espn_summary, sample_game_meta):
        _, plays = parse_game(sample_espn_summary, sample_game_meta)
        required = {"game_id", "play_id", "period", "home_score", "away_score",
                    "score_differential", "home_win_prob", "actual_home_win",
                    "total_seconds_elapsed", "game_clock_pct"}
        for play in plays:
            assert required.issubset(play.keys()), f"Missing fields: {required - play.keys()}"

    def test_score_differential(self, sample_espn_summary, sample_game_meta):
        _, plays = parse_game(sample_espn_summary, sample_game_meta)
        for p in plays:
            assert p["score_differential"] == p["home_score"] - p["away_score"]

    def test_game_clock_pct_in_range(self, sample_espn_summary, sample_game_meta):
        _, plays = parse_game(sample_espn_summary, sample_game_meta)
        for p in plays:
            if p["game_clock_pct"] is not None:
                assert 0.0 <= p["game_clock_pct"] <= 1.0

    def test_weather_extracted(self, sample_espn_summary, sample_game_meta):
        game, _ = parse_game(sample_espn_summary, sample_game_meta)
        assert game["weather_temp_f"] == 72.0
        assert game["weather_condition"] == "Partly Cloudy"

    def test_spread_extracted(self, sample_espn_summary, sample_game_meta):
        game, _ = parse_game(sample_espn_summary, sample_game_meta)
        assert game["espn_spread"] == -17.5
        assert game["espn_over_under"] == 49.5

    def test_attendance(self, sample_espn_summary, sample_game_meta):
        game, _ = parse_game(sample_espn_summary, sample_game_meta)
        assert game["attendance"] == 93246

    def test_no_winprobability_returns_empty_plays(self, sample_espn_summary, sample_game_meta):
        raw = dict(sample_espn_summary)
        raw["winprobability"] = []
        _, plays = parse_game(raw, sample_game_meta)
        assert plays == []

    def test_meta_conference_attached(self, sample_espn_summary, sample_game_meta):
        game, _ = parse_game(sample_espn_summary, sample_game_meta)
        assert game["home_conference"] == "9"
        assert game["group"] == 80

    def test_overtime_detection(self, sample_espn_summary, sample_game_meta):
        # Add an OT play
        ot_play = {
            "id": "1005",
            "period": {"number": 5},
            "clock": {"displayValue": "10:00", "value": 600.0},
            "homeScore": 30,
            "awayScore": 30,
            "type": {"text": "Rush"},
            "scoringPlay": False,
            "start": {"down": 1, "distance": 10, "yardLine": 50, "yardsToEndzone": 50},
            "end": {"yardsToEndzone": 40},
            "statYardage": 10,
            "text": "OT play",
        }
        raw = dict(sample_espn_summary)
        raw["winprobability"] = raw["winprobability"] + [
            {"playId": "1005", "homeWinPercentage": 55.0, "tiePercentage": 0.0}
        ]
        raw["drives"]["previous"][0]["plays"].append(ot_play)
        game, plays = parse_game(raw, sample_game_meta)
        ot_plays = [p for p in plays if p["is_overtime"]]
        assert len(ot_plays) == 1
        assert game["went_to_overtime"] is True


class TestTimeHelpers:
    def test_start_of_first_quarter(self):
        # Period 1, 15:00 remaining → 0 seconds elapsed
        elapsed = _compute_seconds_elapsed(1, 900.0)
        assert elapsed == 0.0

    def test_end_of_first_quarter(self):
        # Period 1, 0:00 remaining → 900 seconds elapsed
        elapsed = _compute_seconds_elapsed(1, 0.0)
        assert elapsed == 900.0

    def test_halftime(self):
        # Period 2, 0:00 remaining → 1800 seconds elapsed
        elapsed = _compute_seconds_elapsed(2, 0.0)
        assert elapsed == 1800.0

    def test_end_of_regulation(self):
        # Period 4, 0:00 remaining → 3600 seconds elapsed
        elapsed = _compute_seconds_elapsed(4, 0.0)
        assert elapsed == _REGULATION_SECONDS

    def test_overtime_exceeds_regulation(self):
        # OT period 5, 5:00 remaining (halfway through OT) → > 3600
        elapsed = _compute_seconds_elapsed(5, 300.0)
        assert elapsed > _REGULATION_SECONDS

    def test_game_clock_pct_halftime(self):
        elapsed = _compute_seconds_elapsed(2, 0.0)
        pct = _game_clock_pct(elapsed)
        assert abs(pct - 0.5) < 1e-6

    def test_game_clock_pct_ot_capped_at_one(self):
        elapsed = _compute_seconds_elapsed(5, 600.0)
        pct = _game_clock_pct(elapsed)
        assert pct == 1.0
