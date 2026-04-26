"""
Parse raw ESPN game summary JSON into structured play-level and game-level records.

The ESPN summary endpoint returns two parallel arrays that we join on playId:
  - `winprobability`: [{ playId, homeWinPercentage, tiePercentage }, ...]
  - `plays` (inside `drives`): [{ id, period, clock, homeScore, awayScore,
                                   down, distance, yardLine, type, ... }, ...]

We also extract game-level metadata (weather, odds, attendance, teams, etc.)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Seconds per regulation quarter
_QUARTER_SECONDS = 15 * 60  # 15 minutes
_REGULATION_SECONDS = 4 * _QUARTER_SECONDS  # 3600 seconds


def parse_game(raw: dict, game_meta: dict | None = None) -> tuple[dict, list[dict]]:
    """
    Parse a raw ESPN summary JSON into a game record and a list of play records.

    Args:
        raw: The full ESPN summary JSON for a game.
        game_meta: Optional metadata dict from the schedule scraper (adds conference, etc.).

    Returns:
        (game_record, plays)  where:
          game_record — one dict with game-level fields
          plays       — list of dicts, one per play that has a win probability
    """
    game_record = _parse_game_record(raw, game_meta)
    plays = _parse_plays(raw, game_record)
    return game_record, plays


# ---------------------------------------------------------------------------
# Game-level parsing
# ---------------------------------------------------------------------------

def _parse_game_record(raw: dict, meta: dict | None) -> dict:
    header = raw.get("header", {})
    competitions = header.get("competitions", [{}])
    comp = competitions[0] if competitions else {}

    competitors = comp.get("competitors", [])
    home = _find_competitor(competitors, "home")
    away = _find_competitor(competitors, "away")

    # Season / type
    season = header.get("season", {})

    # Weather and attendance from gameInfo
    game_info = raw.get("gameInfo", {})

    # Odds from ESPN (sometimes present)
    predictor = raw.get("predictor", {})
    espn_home_prob_pregame = None
    if predictor:
        home_team_pred = predictor.get("homeTeam", {})
        espn_home_prob_pregame = home_team_pred.get("gameProjection")
        if espn_home_prob_pregame is not None:
            try:
                espn_home_prob_pregame = float(espn_home_prob_pregame) / 100.0
            except (ValueError, TypeError):
                espn_home_prob_pregame = None

    # Odds/lines embedded by ESPN
    pickcenter = raw.get("pickcenter", [])
    spread = None
    over_under = None
    if pickcenter:
        # Use first provider's data
        pc = pickcenter[0]
        try:
            spread = float(pc.get("spread", 0) or 0) or None
        except (TypeError, ValueError):
            pass
        try:
            over_under = float(pc.get("overUnder", 0) or 0) or None
        except (TypeError, ValueError):
            pass

    venue = game_info.get("venue", {})
    weather = game_info.get("weather", {})

    record: dict[str, Any] = {
        "game_id": comp.get("id", header.get("id", "")),
        "season": season.get("year"),
        "season_type": _season_type_name(season.get("type")),
        "season_type_id": season.get("type") if isinstance(season.get("type"), int) else None,
        "week": comp.get("week", {}).get("number") if isinstance(comp.get("week"), dict) else comp.get("week"),
        "date": comp.get("date", ""),
        "home_team": home.get("team", {}).get("displayName", ""),
        "home_team_id": home.get("team", {}).get("id", ""),
        "home_team_abbr": home.get("team", {}).get("abbreviation", ""),
        "home_score_final": _safe_int(home.get("score")),
        "away_team": away.get("team", {}).get("displayName", ""),
        "away_team_id": away.get("team", {}).get("id", ""),
        "away_team_abbr": away.get("team", {}).get("abbreviation", ""),
        "away_score_final": _safe_int(away.get("score")),
        "neutral_site": comp.get("neutralSite", False),
        "attendance": _safe_int(game_info.get("attendance")),
        "venue_name": venue.get("fullName", ""),
        "venue_city": venue.get("address", {}).get("city", ""),
        "venue_state": venue.get("address", {}).get("state", ""),
        "weather_temp_f": _safe_float(weather.get("temperature")),
        "weather_condition": weather.get("displayValue", ""),
        "weather_wind_speed": _safe_float(weather.get("windSpeed")),
        "weather_wind_direction": weather.get("windDirection", ""),
        "weather_humidity": _safe_float(weather.get("humidity")),
        "espn_home_prob_pregame": espn_home_prob_pregame,
        "espn_spread": spread,
        "espn_over_under": over_under,
        # From schedule meta (enriched downstream)
        "home_conference": meta.get("home_conference") if meta else None,
        "away_conference": meta.get("away_conference") if meta else None,
        "group": meta.get("group") if meta else None,  # 80=FBS, 81=FCS
    }

    # Determine actual winner
    if record["home_score_final"] is not None and record["away_score_final"] is not None:
        record["actual_home_win"] = int(record["home_score_final"] > record["away_score_final"])
        record["went_to_overtime"] = False  # set later when parsing plays
    else:
        record["actual_home_win"] = None
        record["went_to_overtime"] = False

    return record


def _find_competitor(competitors: list[dict], home_away: str) -> dict:
    for c in competitors:
        if c.get("homeAway") == home_away:
            return c
    return {}


# ---------------------------------------------------------------------------
# Play-level parsing
# ---------------------------------------------------------------------------

def _parse_plays(raw: dict, game_record: dict) -> list[dict]:
    """Build a list of play dicts joined with win probability."""
    # Build win probability lookup: playId → homeWinPercentage
    # ESPN returns values on 0-1 scale (e.g. 0.625), NOT 0-100
    wp_lookup: dict[str, float] = {}
    for wp_entry in raw.get("winprobability", []):
        pid = str(wp_entry.get("playId", ""))
        pct = wp_entry.get("homeWinPercentage")
        if pid and pct is not None:
            try:
                val = float(pct)
                # Normalise: values > 1.0 are on 0-100 scale (older API responses)
                wp_lookup[pid] = val / 100.0 if val > 1.0 else val
            except (TypeError, ValueError):
                pass

    if not wp_lookup:
        # No win probability data for this game
        return []

    # Collect plays from drives
    plays_raw = []
    for drive in raw.get("drives", {}).get("previous", []):
        for play in drive.get("plays", []):
            plays_raw.append(play)

    # Also check top-level plays key (some ESPN responses use this)
    if not plays_raw:
        plays_raw = raw.get("plays", [])

    game_id = game_record["game_id"]
    season = game_record["season"]
    actual_home_win = game_record["actual_home_win"]
    home_score_final = game_record["home_score_final"]
    away_score_final = game_record["away_score_final"]

    plays = []
    max_period = 0

    for play in plays_raw:
        pid = str(play.get("id", ""))
        if pid not in wp_lookup:
            continue

        period = _safe_int(play.get("period", {}).get("number") if isinstance(play.get("period"), dict) else play.get("period")) or 0
        if period > max_period:
            max_period = period

        clock_val = play.get("clock", {})
        if isinstance(clock_val, dict):
            clock_display = clock_val.get("displayValue", "")
            # ESPN sometimes provides "value" (seconds remaining); if absent, parse displayValue
            raw_val = clock_val.get("value")
            if raw_val is not None:
                clock_seconds_remaining = _safe_float(raw_val)
            else:
                clock_seconds_remaining = _parse_clock_display(clock_display)
        else:
            clock_seconds_remaining = None
            clock_display = ""

        total_seconds_elapsed = _compute_seconds_elapsed(period, clock_seconds_remaining)

        home_score = _safe_int(play.get("homeScore")) or 0
        away_score = _safe_int(play.get("awayScore")) or 0

        start_info = play.get("start", {}) or {}
        end_info = play.get("end", {}) or {}

        play_type = ""
        if isinstance(play.get("type"), dict):
            play_type = play["type"].get("text", play["type"].get("abbreviation", ""))

        rec: dict[str, Any] = {
            "game_id": game_id,
            "play_id": pid,
            "season": season,
            "period": period,
            "is_overtime": period > 4,
            "clock_display": clock_display,
            "clock_seconds_remaining_period": clock_seconds_remaining,
            "total_seconds_elapsed": total_seconds_elapsed,
            "game_clock_pct": _game_clock_pct(total_seconds_elapsed),
            "home_score": home_score,
            "away_score": away_score,
            "score_differential": home_score - away_score,
            "down": _safe_int(start_info.get("down")),
            "distance": _safe_int(start_info.get("distance")),
            "yard_line": _safe_int(start_info.get("yardLine")),
            "yards_to_endzone": _safe_int(start_info.get("yardsToEndzone")),
            "yards_gained": _safe_int(play.get("statYardage")),
            "play_type": play_type,
            "play_description": play.get("text", "")[:300],
            "is_scoring_play": bool(play.get("scoringPlay", False)),
            "home_win_prob": wp_lookup[pid],
            # Ground truth — same for every play in the game
            "actual_home_win": actual_home_win,
            "home_score_final": home_score_final,
            "away_score_final": away_score_final,
        }
        plays.append(rec)

    # Flag overtime
    if max_period > 4:
        game_record["went_to_overtime"] = True

    return plays


def _compute_seconds_elapsed(period: int | None, seconds_remaining: float | None) -> float | None:
    """Convert (period, clock_seconds_remaining) to total seconds elapsed in regulation."""
    if period is None:
        return None
    if period <= 4:
        elapsed_full_quarters = (period - 1) * _QUARTER_SECONDS
        if seconds_remaining is not None:
            return elapsed_full_quarters + (_QUARTER_SECONDS - seconds_remaining)
        return elapsed_full_quarters
    else:
        # Overtime: return regulation length + some positive offset
        # We encode OT as 3600 + (ot_period - 1) * 600 + (600 - remaining)
        ot_period = period - 4
        ot_quarter_seconds = 10 * 60  # 10-minute OT periods
        elapsed_ot = (ot_period - 1) * ot_quarter_seconds
        if seconds_remaining is not None:
            elapsed_ot += ot_quarter_seconds - seconds_remaining
        return _REGULATION_SECONDS + elapsed_ot


def _game_clock_pct(total_seconds_elapsed: float | None) -> float | None:
    """Return fraction of regulation elapsed, capped at 1.0 for OT plays."""
    if total_seconds_elapsed is None:
        return None
    return min(total_seconds_elapsed / _REGULATION_SECONDS, 1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEASON_TYPE_MAP = {
    1: "preseason",
    2: "regular",
    3: "postseason",
    4: "offseason",
}


def _season_type_name(val: Any) -> str:
    """Map ESPN season type int/dict to a human-readable string."""
    if isinstance(val, int):
        return _SEASON_TYPE_MAP.get(val, str(val))
    if isinstance(val, dict):
        return val.get("name", val.get("slug", ""))
    return str(val) if val is not None else ""


def _parse_clock_display(display: str) -> float | None:
    """Parse a MM:SS or M:SS clock string to seconds remaining."""
    if not display:
        return None
    try:
        parts = display.strip().split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
    except (ValueError, IndexError):
        pass
    return None


def _safe_int(val: Any) -> int | None:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
