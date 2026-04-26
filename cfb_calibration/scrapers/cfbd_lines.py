"""
College Football Data API (CFBD) scraper for Vegas lines.

Fetches pre-game spread, over/under, and moneyline for all games in a season.
Requires a free API key from https://collegefootballdata.com/key

Set CFBD_API_KEY in your .env file or environment variable.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

CFBD_BASE_URL = "https://api.collegefootballdata.com"
_TIMEOUT = 30.0


def _get_api_key() -> str | None:
    """Load CFBD API key from environment or .env file."""
    key = os.getenv("CFBD_API_KEY")
    if not key:
        # Search for .env walking up from this file's location to the project root
        search = Path(__file__).resolve().parent
        for _ in range(5):
            env_path = search / ".env"
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("CFBD_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
            if key:
                break
            search = search.parent
    return key


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _fetch_lines_sync(api_key: str, year: int, season_type: str = "regular") -> list[dict]:
    """Fetch betting lines for a season synchronously."""
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"year": year, "seasonType": season_type}
    with httpx.Client(headers=headers, timeout=_TIMEOUT) as client:
        resp = client.get(f"{CFBD_BASE_URL}/lines", params=params)
        resp.raise_for_status()
    return resp.json()


def collect_lines(
    season: int,
    output_dir: Path,
    *,
    season_types: list[str] | None = None,
) -> list[dict]:
    """
    Fetch Vegas lines for a season from CFBD and save to output_dir/{season}.json.

    If CFBD_API_KEY is not set, logs a warning and returns an empty list.

    Args:
        season: Year (e.g. 2023).
        output_dir: Directory to save JSON files.
        season_types: List of season types to fetch. Defaults to ['regular', 'postseason'].

    Returns:
        List of game line dicts.
    """
    if season_types is None:
        season_types = ["regular", "postseason"]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{season}.json"

    if out_path.exists():
        logger.info("Lines for %d already exist at %s, loading from disk.", season, out_path)
        return json.loads(out_path.read_text())

    api_key = _get_api_key()
    if not api_key:
        logger.warning(
            "CFBD_API_KEY not set. Skipping Vegas lines for season %d. "
            "Get a free key at https://collegefootballdata.com/key and set it in .env.",
            season,
        )
        return []

    all_lines: dict[str, dict] = {}
    for season_type in season_types:
        try:
            lines = _fetch_lines_sync(api_key, season, season_type)
            for game in lines:
                gid = str(game.get("id", ""))
                if gid:
                    all_lines[gid] = _normalize_line(game)
        except Exception as exc:
            logger.warning("Failed to fetch %s lines for %d: %s", season_type, season, exc)

    result = list(all_lines.values())
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Season %d: saved %d line records to %s", season, len(result), out_path)
    return result


def _normalize_line(raw: dict) -> dict:
    """Extract the most useful fields from a CFBD lines record."""
    # CFBD returns multiple provider lines; pick the consensus / first available
    lines = raw.get("lines", [])
    spread = None
    over_under = None
    home_moneyline = None
    away_moneyline = None

    for line in lines:
        if spread is None and line.get("spread") not in (None, ""):
            try:
                spread = float(line["spread"])
            except (TypeError, ValueError):
                pass
        if over_under is None and line.get("overUnder") not in (None, ""):
            try:
                over_under = float(line["overUnder"])
            except (TypeError, ValueError):
                pass
        if home_moneyline is None and line.get("homeMoneyline") not in (None, ""):
            try:
                home_moneyline = int(line["homeMoneyline"])
            except (TypeError, ValueError):
                pass
        if away_moneyline is None and line.get("awayMoneyline") not in (None, ""):
            try:
                away_moneyline = int(line["awayMoneyline"])
            except (TypeError, ValueError):
                pass

    return {
        "cfbd_game_id": str(raw.get("id", "")),
        "season": raw.get("season"),
        "season_type": raw.get("seasonType", ""),
        "week": raw.get("week"),
        "home_team": raw.get("homeTeam", ""),
        "away_team": raw.get("awayTeam", ""),
        "home_score": raw.get("homeScore"),
        "away_score": raw.get("awayScore"),
        "spread": spread,          # negative = home favored by |spread|
        "over_under": over_under,
        "home_moneyline": home_moneyline,
        "away_moneyline": away_moneyline,
        "start_date": raw.get("startDate", ""),
    }


def build_lines_lookup(lines: list[dict]) -> dict[tuple[str, str, str], dict]:
    """
    Build a lookup dict keyed by (home_team_normalized, away_team_normalized, date_str).
    Used to join CFBD lines onto ESPN play data.
    """
    lookup = {}
    for line in lines:
        home = _normalize_team_name(line["home_team"])
        away = _normalize_team_name(line["away_team"])
        date_str = line["start_date"][:10] if line["start_date"] else ""
        lookup[(home, away, date_str)] = line
    return lookup


def _normalize_team_name(name: str) -> str:
    """Lowercase and strip punctuation for fuzzy team-name matching."""
    import re
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()
