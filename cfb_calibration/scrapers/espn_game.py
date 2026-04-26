"""
ESPN game scraper.

For each game_id, fetches the full summary JSON from ESPN's unofficial API and
saves it as a gzipped JSON file. The summary contains:
  - play-by-play array (down, distance, field position, clock, score, play type, ...)
  - win probability array (homeWinPercentage keyed to playId)
  - game-level metadata (weather, odds, boxscore header, ...)
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
from pathlib import Path

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm

logger = logging.getLogger(__name__)

SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary"
)

_SEMAPHORE_SIZE = 8
_REQUEST_DELAY = 0.3  # seconds


@retry(
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(4),
    reraise=True,
)
async def _fetch_game_summary(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    game_id: str,
) -> dict:
    async with sem:
        await asyncio.sleep(_REQUEST_DELAY)
        resp = await client.get(SUMMARY_URL, params={"event": game_id}, timeout=30.0)
        resp.raise_for_status()
    return resp.json()


def _game_path(output_dir: Path, season: int, game_id: str) -> Path:
    season_dir = output_dir / str(season)
    season_dir.mkdir(parents=True, exist_ok=True)
    return season_dir / f"{game_id}.json.gz"


def _save_game(path: Path, data: dict) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def load_game(path: Path) -> dict:
    """Load a saved game JSON.gz file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


async def collect_games(
    game_records: list[dict],
    output_dir: Path,
    *,
    season: int,
    max_errors: int = 50,
) -> dict[str, str]:
    """
    Fetch and save raw game summaries for a list of game records.

    Args:
        game_records: List of game metadata dicts (must contain 'game_id').
        output_dir: Root directory; files saved as output_dir/{season}/{game_id}.json.gz
        season: Season year (used for partitioning output).
        max_errors: Stop early if this many consecutive errors occur.

    Returns:
        Dict mapping game_id → status ('saved', 'skipped', 'error').
    """
    statuses: dict[str, str] = {}
    to_fetch = []

    for rec in game_records:
        gid = str(rec["game_id"])
        path = _game_path(output_dir, season, gid)
        if path.exists():
            statuses[gid] = "skipped"
        else:
            to_fetch.append(gid)

    logger.info(
        "Season %d: %d games to fetch, %d already on disk.",
        season,
        len(to_fetch),
        len(statuses),
    )

    sem = asyncio.Semaphore(_SEMAPHORE_SIZE)
    error_count = 0

    async with httpx.AsyncClient(
        headers={"User-Agent": "cfb-calibration-study/0.1 (academic research)"},
        follow_redirects=True,
        http2=True,
    ) as client:
        tasks = {gid: _fetch_game_summary(client, sem, gid) for gid in to_fetch}

        for gid, coro in tqdm(
            asyncio.as_completed(list(tasks.values())),
            total=len(tasks),
        desc=f"Games {season}",
        ):
            # asyncio.as_completed doesn't return the key, so we zip manually below
            pass

        # Re-run with proper key tracking
        async def fetch_and_save(gid: str) -> tuple[str, str]:
            try:
                data = await _fetch_game_summary(client, sem, gid)
                path = _game_path(output_dir, season, gid)
                _save_game(path, data)
                return gid, "saved"
            except Exception as exc:
                logger.warning("Failed to fetch game %s: %s", gid, exc)
                return gid, "error"

        coros = [fetch_and_save(gid) for gid in to_fetch]
        for future in tqdm(asyncio.as_completed(coros), total=len(coros), desc=f"Games {season}"):
            gid, status = await future
            statuses[gid] = status
            if status == "error":
                error_count += 1
                if error_count >= max_errors:
                    logger.error("Reached max_errors=%d, aborting season %d.", max_errors, season)
                    break

    saved = sum(1 for s in statuses.values() if s == "saved")
    errors = sum(1 for s in statuses.values() if s == "error")
    logger.info("Season %d complete: %d saved, %d skipped, %d errors.", season, saved, len(statuses) - saved - errors, errors)
    return statuses


async def collect_all_games(
    schedules: dict[int, list[dict]],
    output_dir: Path,
) -> dict[int, dict[str, str]]:
    """Collect games for all seasons. Seasons run sequentially."""
    results = {}
    for season, records in sorted(schedules.items()):
        results[season] = await collect_games(records, output_dir, season=season)
    return results
