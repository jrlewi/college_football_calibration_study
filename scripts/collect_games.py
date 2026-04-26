#!/usr/bin/env python3
"""
Collect ESPN play-by-play + win probability data for all games.

Reads game IDs from schedule JSON files and fetches the full summary
for each game, saving gzipped JSON to data/raw/games/{season}/{game_id}.json.gz

Usage:
    # Collect games for all seasons (reads from data/raw/schedules/)
    uv run scripts/collect_games.py --all

    # Collect a specific season
    uv run scripts/collect_games.py --seasons 2023

    # Collect a single game (useful for testing)
    uv run scripts/collect_games.py --game-id 401628333

    # Rebuild processed Parquet after collecting
    uv run scripts/collect_games.py --seasons 2023 --process
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cfb_calibration.scrapers.espn_game import collect_games, _fetch_game_summary, _game_path, _save_game
from cfb_calibration.processing.pipeline import build_dataset

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCHEDULES_DIR = Path("data/raw/schedules")
GAMES_DIR = Path("data/raw/games")
LINES_DIR = Path("data/raw/lines")
PROCESSED_DIR = Path("data/processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ESPN CFB game data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        metavar="YEAR",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Collect all seasons found in data/raw/schedules/",
    )
    group.add_argument(
        "--game-id",
        type=str,
        metavar="ID",
        help="Fetch a single game by ESPN game ID",
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="After collecting, run the processing pipeline to rebuild Parquet files",
    )
    parser.add_argument(
        "--schedules-dir",
        type=Path,
        default=SCHEDULES_DIR,
    )
    parser.add_argument(
        "--games-dir",
        type=Path,
        default=GAMES_DIR,
    )
    return parser.parse_args()


def load_schedule(season: int, schedules_dir: Path) -> list[dict]:
    path = schedules_dir / f"{season}.json"
    if not path.exists():
        logger.warning("No schedule file for season %d at %s", season, path)
        return []
    return json.loads(path.read_text())


async def fetch_single_game(game_id: str, games_dir: Path) -> None:
    """Fetch and save a single game by ID (for testing/debugging)."""
    # Determine season from game file if possible, default to 0
    out_path = games_dir / "single" / f"{game_id}.json.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Game {game_id} already saved at {out_path}")
        return

    import asyncio
    sem = asyncio.Semaphore(1)
    async with httpx.AsyncClient(
        headers={"User-Agent": "cfb-calibration-study/0.1 (academic research)"},
        follow_redirects=True,
        http2=True,
    ) as client:
        data = await _fetch_game_summary(client, sem, game_id)

    _save_game(out_path, data)
    print(f"Saved game {game_id} to {out_path}")

    # Quick parse check
    from cfb_calibration.processing.parsers import parse_game
    game_rec, plays = parse_game(data)
    print(f"  Teams: {game_rec.get('away_team')} @ {game_rec.get('home_team')}")
    print(f"  Date: {game_rec.get('date', '')[:10]}")
    print(f"  Final: {game_rec.get('away_score_final')}–{game_rec.get('home_score_final')}")
    print(f"  Plays with win probability: {len(plays)}")
    if plays:
        print(f"  Win prob range: {min(p['home_win_prob'] for p in plays):.3f} – {max(p['home_win_prob'] for p in plays):.3f}")


async def main() -> None:
    args = parse_args()

    if args.game_id:
        await fetch_single_game(args.game_id, args.games_dir)
        return

    # Determine seasons
    if args.all:
        seasons = sorted(int(p.stem) for p in args.schedules_dir.glob("*.json"))
        if not seasons:
            print(f"No schedule files found in {args.schedules_dir}. Run collect_schedule.py first.")
            return
    else:
        seasons = args.seasons

    print(f"Collecting game data for seasons: {seasons}")

    for season in seasons:
        records = load_schedule(season, args.schedules_dir)
        if not records:
            continue
        print(f"\nSeason {season}: {len(records)} games in schedule")
        statuses = await collect_games(records, args.games_dir, season=season)
        saved = sum(1 for s in statuses.values() if s == "saved")
        skipped = sum(1 for s in statuses.values() if s == "skipped")
        errors = sum(1 for s in statuses.values() if s == "error")
        print(f"  Saved: {saved}, Skipped: {skipped}, Errors: {errors}")

    if args.process:
        print("\nRunning processing pipeline...")
        build_dataset(
            raw_games_dir=args.games_dir,
            schedules_dir=args.schedules_dir,
            lines_dir=LINES_DIR,
            output_dir=PROCESSED_DIR,
            seasons=seasons if not args.all else None,
        )
        print(f"Processed data written to {PROCESSED_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
