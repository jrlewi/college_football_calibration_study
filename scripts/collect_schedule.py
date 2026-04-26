#!/usr/bin/env python3
"""
Collect ESPN college football game IDs for a range of seasons.

Usage:
    uv run scripts/collect_schedule.py --seasons 2022 2023
    uv run scripts/collect_schedule.py --seasons 2015 2016 2017 --dry-run
    uv run scripts/collect_schedule.py --all          # 2015–2025
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from cfb_calibration.scrapers.espn_schedule import collect_schedules

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ESPN CFB schedule data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        metavar="YEAR",
        help="Season year(s) to collect (e.g. 2022 2023)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Collect all seasons from 2015 to 2025",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/schedules"),
        help="Directory to save schedule JSON files (default: data/raw/schedules)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be fetched without making requests",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    seasons = args.seasons if args.seasons else list(range(2015, 2026))
    output_dir = args.output_dir

    print(f"Collecting schedules for seasons: {seasons}")
    print(f"Output directory: {output_dir}")
    if args.dry_run:
        print("[DRY RUN] No requests will be made.")

    results = await collect_schedules(seasons, output_dir, dry_run=args.dry_run)

    if not args.dry_run:
        total_games = sum(len(games) for games in results.values())
        print(f"\nDone. Total games collected: {total_games:,}")
        for season, games in sorted(results.items()):
            print(f"  {season}: {len(games):,} games")


if __name__ == "__main__":
    asyncio.run(main())
