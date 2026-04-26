"""
End-to-end pipeline: raw game JSON files → processed Parquet files.

Usage:
    from cfb_calibration.processing.pipeline import build_dataset
    build_dataset(
        raw_games_dir=Path("data/raw/games"),
        schedules_dir=Path("data/raw/schedules"),
        lines_dir=Path("data/raw/lines"),
        output_dir=Path("data/processed"),
        seasons=[2022, 2023],
    )
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .features import add_features, add_spread_features
from .parsers import parse_game
from ..scrapers.espn_game import load_game
from ..scrapers.cfbd_lines import build_lines_lookup, _normalize_team_name

logger = logging.getLogger(__name__)


def build_dataset(
    raw_games_dir: Path,
    schedules_dir: Path,
    lines_dir: Path,
    output_dir: Path,
    seasons: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse all saved game JSON files and write plays.parquet and games.parquet.

    Returns:
        (games_df, plays_df)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load schedules for metadata enrichment
    schedule_meta: dict[str, dict] = {}
    for sched_file in sorted(schedules_dir.glob("*.json")):
        records = json.loads(sched_file.read_text())
        for rec in records:
            schedule_meta[str(rec["game_id"])] = rec

    # Load lines lookups by season
    lines_by_season: dict[int, dict] = {}
    for lines_file in sorted(lines_dir.glob("*.json")):
        season_year = int(lines_file.stem)
        if seasons and season_year not in seasons:
            continue
        records = json.loads(lines_file.read_text())
        lines_by_season[season_year] = build_lines_lookup(records)

    all_games: list[dict] = []
    all_plays: list[dict] = []

    # Iterate game JSON files
    season_dirs = sorted(raw_games_dir.iterdir()) if raw_games_dir.exists() else []
    for season_dir in season_dirs:
        if not season_dir.is_dir():
            continue
        try:
            season = int(season_dir.name)
        except ValueError:
            continue
        if seasons and season not in seasons:
            continue

        lines_lookup = lines_by_season.get(season, {})
        game_files = list(season_dir.glob("*.json.gz"))
        logger.info("Processing season %d: %d game files", season, len(game_files))

        for gf in tqdm(game_files, desc=f"Parsing {season}"):
            game_id = gf.stem.replace(".json", "")
            meta = schedule_meta.get(game_id)
            try:
                raw = load_game(gf)
                game_rec, plays = parse_game(raw, meta)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", gf.name, exc)
                continue

            if not plays:
                continue

            # Enrich game record with Vegas lines
            _enrich_with_lines(game_rec, lines_lookup)

            # Attach game-level fields to each play
            game_fields = {
                "season": game_rec["season"],
                "season_type": game_rec["season_type"],
                "week": game_rec["week"],
                "home_team": game_rec["home_team"],
                "away_team": game_rec["away_team"],
                "home_conference": game_rec["home_conference"],
                "away_conference": game_rec["away_conference"],
                "group": game_rec["group"],
                "neutral_site": game_rec["neutral_site"],
                "attendance": game_rec["attendance"],
                "actual_home_win": game_rec["actual_home_win"],
                "went_to_overtime": game_rec["went_to_overtime"],
                "pre_game_spread": game_rec.get("pre_game_spread"),
                "over_under": game_rec.get("over_under"),
                "weather_temp_f": game_rec.get("weather_temp_f"),
                "weather_condition": game_rec.get("weather_condition"),
            }
            for play in plays:
                play.update(game_fields)

            all_games.append(game_rec)
            all_plays.extend(plays)

    if not all_games:
        logger.warning("No games parsed. Check raw_games_dir and seasons filter.")
        return pd.DataFrame(), pd.DataFrame()

    games_df = pd.DataFrame(all_games)
    plays_df = pd.DataFrame(all_plays)

    # Add engineered features
    plays_df = add_features(plays_df)
    plays_df = add_spread_features(plays_df)

    # Write output
    games_path = output_dir / "games.parquet"
    plays_path = output_dir / "plays.parquet"
    games_df.to_parquet(games_path, index=False)
    plays_df.to_parquet(plays_path, index=False)

    logger.info(
        "Wrote %d games and %d plays to %s",
        len(games_df),
        len(plays_df),
        output_dir,
    )
    return games_df, plays_df


def _enrich_with_lines(game_rec: dict, lines_lookup: dict) -> None:
    """Try to match the game to a CFBD lines record and add spread/over-under."""
    if not lines_lookup:
        return

    home = _normalize_team_name(game_rec.get("home_team", ""))
    away = _normalize_team_name(game_rec.get("away_team", ""))
    date_str = game_rec.get("date", "")[:10]

    line = lines_lookup.get((home, away, date_str))
    if line is None:
        # Try reverse (neutral site games may have teams swapped in CFBD)
        line = lines_lookup.get((away, home, date_str))
        if line is not None:
            # Flip the spread sign
            line = dict(line)
            if line.get("spread") is not None:
                line["spread"] = -line["spread"]

    if line:
        game_rec["pre_game_spread"] = line.get("spread")
        game_rec["over_under"] = line.get("over_under")
        game_rec["home_moneyline"] = line.get("home_moneyline")
        game_rec["away_moneyline"] = line.get("away_moneyline")
    else:
        game_rec.setdefault("pre_game_spread", game_rec.get("espn_spread"))
        game_rec.setdefault("over_under", game_rec.get("espn_over_under"))
