"""
Shared pytest fixtures.

Provides minimal but realistic ESPN summary JSON for parser tests,
and synthetic probability arrays for calibration metric tests.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# ESPN game summary fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_espn_summary() -> dict:
    """
    Minimal ESPN game summary JSON structure with 4 plays and win probabilities.
    Mirrors the real ESPN API response shape.
    """
    return {
        "header": {
            "id": "401628333",
            "season": {"year": 2023, "type": {"id": 2, "name": "Regular Season"}, "slug": "regular-season"},
            "week": {"number": 5},
            "competitions": [
                {
                    "id": "401628333",
                    "date": "2023-10-07T19:00Z",
                    "neutralSite": False,
                    "attendance": 93246,
                    "venue": {
                        "fullName": "Sanford Stadium",
                        "address": {"city": "Athens", "state": "GA"},
                    },
                    "competitors": [
                        {
                            "homeAway": "home",
                            "score": "30",
                            "team": {
                                "id": "61",
                                "displayName": "Georgia Bulldogs",
                                "abbreviation": "UGA",
                                "conferenceId": "9",
                            },
                        },
                        {
                            "homeAway": "away",
                            "score": "13",
                            "team": {
                                "id": "333",
                                "displayName": "Auburn Tigers",
                                "abbreviation": "AUB",
                                "conferenceId": "9",
                            },
                        },
                    ],
                }
            ],
        },
        "gameInfo": {
            "attendance": 93246,
            "venue": {
                "fullName": "Sanford Stadium",
                "address": {"city": "Athens", "state": "GA"},
            },
            "weather": {
                "temperature": 72.0,
                "displayValue": "Partly Cloudy",
                "windSpeed": 8.0,
                "windDirection": "SW",
                "humidity": 55.0,
            },
        },
        "pickcenter": [
            {
                "spread": -17.5,
                "overUnder": 49.5,
                "provider": {"name": "ESPN BET"},
            }
        ],
        "winprobability": [
            {"playId": "1001", "homeWinPercentage": 62.5, "tiePercentage": 0.0},
            {"playId": "1002", "homeWinPercentage": 70.0, "tiePercentage": 0.0},
            {"playId": "1003", "homeWinPercentage": 85.0, "tiePercentage": 0.0},
            {"playId": "1004", "homeWinPercentage": 95.0, "tiePercentage": 0.0},
        ],
        "drives": {
            "previous": [
                {
                    "plays": [
                        {
                            "id": "1001",
                            "period": {"number": 1},
                            "clock": {"displayValue": "12:30", "value": 750.0},
                            "homeScore": 0,
                            "awayScore": 0,
                            "type": {"text": "Rush", "abbreviation": "RUSH"},
                            "scoringPlay": False,
                            "start": {"down": 1, "distance": 10, "yardLine": 75, "yardsToEndzone": 75},
                            "end": {"yardsToEndzone": 70},
                            "statYardage": 5,
                            "text": "QB rush for 5 yards",
                        },
                        {
                            "id": "1002",
                            "period": {"number": 2},
                            "clock": {"displayValue": "8:15", "value": 495.0},
                            "homeScore": 7,
                            "awayScore": 0,
                            "type": {"text": "Pass", "abbreviation": "PASS"},
                            "scoringPlay": False,
                            "start": {"down": 2, "distance": 5, "yardLine": 45, "yardsToEndzone": 45},
                            "end": {"yardsToEndzone": 35},
                            "statYardage": 10,
                            "text": "Pass complete for 10 yards",
                        },
                        {
                            "id": "1003",
                            "period": {"number": 3},
                            "clock": {"displayValue": "5:00", "value": 300.0},
                            "homeScore": 21,
                            "awayScore": 7,
                            "type": {"text": "Pass", "abbreviation": "PASS"},
                            "scoringPlay": False,
                            "start": {"down": 1, "distance": 10, "yardLine": 60, "yardsToEndzone": 60},
                            "end": {"yardsToEndzone": 50},
                            "statYardage": 10,
                            "text": "Pass complete for 10 yards",
                        },
                        {
                            "id": "1004",
                            "period": {"number": 4},
                            "clock": {"displayValue": "2:00", "value": 120.0},
                            "homeScore": 30,
                            "awayScore": 13,
                            "type": {"text": "Rush", "abbreviation": "RUSH"},
                            "scoringPlay": False,
                            "start": {"down": 1, "distance": 10, "yardLine": 30, "yardsToEndzone": 30},
                            "end": {"yardsToEndzone": 25},
                            "statYardage": 5,
                            "text": "Rush for 5 yards",
                        },
                    ]
                }
            ]
        },
    }


@pytest.fixture
def sample_game_meta() -> dict:
    return {
        "game_id": "401628333",
        "home_conference": "9",
        "away_conference": "9",
        "group": 80,
        "season_type": "regular-season",
    }


# ---------------------------------------------------------------------------
# Synthetic probability arrays
# ---------------------------------------------------------------------------

@pytest.fixture
def perfectly_calibrated():
    """Synthetic perfectly calibrated predictions."""
    rng = np.random.default_rng(42)
    probs = rng.uniform(0, 1, 10_000)
    outcomes = (rng.uniform(0, 1, 10_000) < probs).astype(float)
    return probs, outcomes


@pytest.fixture
def overconfident():
    """Overconfident predictions: model predicts 0.8 when true rate is ~0.6."""
    rng = np.random.default_rng(42)
    probs = rng.beta(5, 1.5, 5_000)  # skewed high
    outcomes = (rng.uniform(0, 1, 5_000) < (probs * 0.75)).astype(float)
    return probs, outcomes


@pytest.fixture
def all_fifty_fifty():
    """Edge case: all predictions are exactly 0.5."""
    n = 1000
    probs = np.full(n, 0.5)
    rng = np.random.default_rng(42)
    outcomes = rng.integers(0, 2, n).astype(float)
    return probs, outcomes
