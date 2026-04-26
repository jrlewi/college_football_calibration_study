"""
ESPN schedule scraper.

Collects all college football game IDs for a range of seasons by iterating
through every date in the college football calendar (late August – late January)
and querying ESPN's scoreboard endpoint for both FBS (groups=80) and FCS (groups=81).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)

# ESPN scoreboard endpoint
SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
)

# Group IDs: 80 = FBS, 81 = FCS
GROUPS = [80, 81]

# College football season runs roughly Aug 24 – Jan 31
SEASON_START_MONTH_DAY = (8, 24)
SEASON_END_MONTH_DAY = (1, 31)

# Concurrent requests and polite delay
_SEMAPHORE_SIZE = 8
_REQUEST_DELAY = 0.3  # seconds between requests within a batch


def season_dates(season: int) -> Iterator[date]:
    """Yield every date in the college football season for a given year.

    A 'season' year (e.g. 2023) covers:
        Aug 24, 2023 → Jan 31, 2024
    """
    start = date(season, *SEASON_START_MONTH_DAY)
    end = date(season + 1, *SEASON_END_MONTH_DAY)
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
async def _fetch_scoreboard(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    date_str: str,
    group: int,
) -> list[dict]:
    """Fetch scoreboard for one date + group, return list of game dicts."""
    params = {"dates": date_str, "groups": group, "limit": 500}
    async with sem:
        await asyncio.sleep(_REQUEST_DELAY)
        resp = await client.get(SCOREBOARD_URL, params=params, timeout=20.0)
        resp.raise_for_status()

    data = resp.json()
    events = data.get("events", [])
    games = []
    for event in events:
        try:
            comp = event["competitions"][0]
            home = next(t for t in comp["competitors"] if t["homeAway"] == "home")
            away = next(t for t in comp["competitors"] if t["homeAway"] == "away")
            games.append(
                {
                    "game_id": event["id"],
                    "date": event.get("date", ""),
                    "season": event.get("season", {}).get("year"),
                    "season_type": event.get("season", {}).get("slug", ""),
                    "week": event.get("week", {}).get("number"),
                    "home_team": home["team"]["displayName"],
                    "home_team_id": home["team"]["id"],
                    "home_team_abbr": home["team"].get("abbreviation", ""),
                    "home_conference": home["team"].get("conferenceId", ""),
                    "away_team": away["team"]["displayName"],
                    "away_team_id": away["team"]["id"],
                    "away_team_abbr": away["team"].get("abbreviation", ""),
                    "away_conference": away["team"].get("conferenceId", ""),
                    "neutral_site": comp.get("neutralSite", False),
                    "attendance": comp.get("attendance"),
                    "venue_name": comp.get("venue", {}).get("fullName", ""),
                    "venue_city": comp.get("venue", {}).get("address", {}).get("city", ""),
                    "venue_state": comp.get("venue", {}).get("address", {}).get("state", ""),
                    "status": event.get("status", {}).get("type", {}).get("name", ""),
                    "group": group,
                }
            )
        except (KeyError, StopIteration) as exc:
            logger.debug("Skipping malformed event %s: %s", event.get("id"), exc)
    return games


async def collect_season_schedule(
    season: int,
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> list[dict]:
    """
    Collect all game IDs for a season and save to output_dir/{season}.json.

    Returns the list of game metadata dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{season}.json"

    if out_path.exists():
        logger.info("Schedule for %d already exists at %s, loading from disk.", season, out_path)
        return json.loads(out_path.read_text())

    dates = list(season_dates(season))
    all_games: dict[str, dict] = {}  # game_id → game dict (dedup)

    sem = asyncio.Semaphore(_SEMAPHORE_SIZE)

    n_combos = len(dates) * len(GROUPS)
    if dry_run:
        logger.info("[dry-run] Would fetch %d date×group combos for season %d", n_combos, season)
        return []

    async with httpx.AsyncClient(
        headers={"User-Agent": "cfb-calibration-study/0.1 (academic research)"},
        follow_redirects=True,
        http2=True,
    ) as client:
        tasks = [
            _fetch_scoreboard(client, sem, d.strftime("%Y%m%d"), group)
            for d in dates
            for group in GROUPS
        ]
        desc = f"Schedule {season}"

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
            try:
                games = await coro
                for g in games:
                    all_games[g["game_id"]] = g
            except Exception as exc:
                logger.warning("Failed to fetch scoreboard chunk: %s", exc)

    result = list(all_games.values())
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Season %d: saved %d games to %s", season, len(result), out_path)
    return result


async def collect_schedules(
    seasons: list[int],
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> dict[int, list[dict]]:
    """Collect schedules for multiple seasons sequentially (to be kind to ESPN)."""
    results = {}
    for season in seasons:
        results[season] = await collect_season_schedule(season, output_dir, dry_run=dry_run)
    return results
