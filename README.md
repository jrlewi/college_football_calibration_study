# College Football Win Probability Calibration Study

An analysis of how well ESPN's in-game win probability model is calibrated across college football seasons. The study measures whether a predicted 70% win probability actually corresponds to a 70% win rate, and investigates where the model over- or under-estimates confidence — broken down by game state, score differential, down & distance, conference tier, and Vegas spread.

Findings are written up as a blog post in `blog_post_draft.md`.

---

## Data Sources

### ESPN (no key required)
Play-by-play and win probability data come from ESPN's unofficial public API:

- **Scoreboard endpoint** — used to collect all game IDs for a season by iterating over every date in the college football calendar (late August through late January) for both FBS (Group 80) and FCS (Group 81).
- **Game summary endpoint** — for each game ID, fetches the full summary JSON containing the play-by-play array (down, distance, field position, clock, score, play type) and the win probability array (`homeWinPercentage` keyed to `playId`). Files are saved as gzipped JSON under `data/raw/games/`.

No authentication is required for these endpoints. Requests are rate-limited with a polite delay and retried automatically on failure.

### College Football Data API — CFBD (free key required)
Pre-game Vegas lines (spread, over/under, moneyline) come from the [College Football Data API](https://collegefootballdata.com). CFBD aggregates lines from multiple sportsbook providers; this project takes the first available consensus value for each game.

**Getting a CFBD API key:**
1. Go to https://collegefootballdata.com/key
2. Register for a free account
3. Copy your API key into `.env`:

```bash
cp .env.example .env
# Edit .env and set:
CFBD_API_KEY=your_key_here
```

The key is loaded automatically from `.env` at runtime. If it is missing, Vegas lines are skipped with a warning and the analysis proceeds without spread-based segmentation.

---

## Project Structure

```
college_football_calibration_study/
├── cfb_calibration/            # Core Python package
│   ├── scrapers/               # Data collection from external APIs
│   │   ├── espn_schedule.py    # Collects all game IDs for a season range by scraping ESPN's
│   │   │                       #   scoreboard endpoint (FBS + FCS, async with concurrency control)
│   │   ├── espn_game.py        # Downloads full game summaries (play-by-play, win probability,
│   │   │                       #   metadata) from ESPN; saves as gzipped JSON files
│   │   └── cfbd_lines.py       # Fetches pre-game Vegas spreads, over/unders, and moneylines
│   │                           #   from the CFBD API; normalizes across sportsbook providers
│   ├── processing/             # Raw → structured data pipeline
│   │   ├── parsers.py          # Parses raw ESPN JSON: joins win probability array to play-by-play
│   │   │                       #   on playId; extracts game-level metadata (weather, attendance, etc.)
│   │   ├── features.py         # Adds derived columns: score differential bins, game clock bins,
│   │   │                       #   down/distance situations, conference tier (P4/G5/FCS),
│   │   │                       #   spread categories, and close-game flag
│   │   └── pipeline.py         # End-to-end pipeline: reads raw game files, runs parsers +
│   │                           #   features, joins Vegas lines, writes plays.parquet + games.parquet
│   ├── analysis/               # Calibration metrics and segmentation
│   │   ├── calibration.py      # Core metrics: reliability diagram data, Brier score, log loss,
│   │   │                       #   ECE (expected calibration error), MCE, and per-segment summaries
│   │   └── segmentation.py     # 2D grid analysis: computes ECE across combinations of two
│   │                           #   variables (e.g. game clock × score differential heatmap)
│   └── visualization/
│       └── plots.py            # Publication-quality reliability diagrams, ECE heatmaps, and
│                               #   calibration gap charts styled for blog/Medium posts
├── notebooks/                  # Analysis notebooks (run in order)
│   ├── 01_data_collection.ipynb    # Orchestrates scraping and saves raw data
│   ├── 02_eda.ipynb                # Exploratory analysis of play and game distributions
│   └── 03_calibration_analysis.ipynb  # Full calibration analysis and figure generation
├── scripts/                    # Standalone data collection scripts
│   ├── collect_schedule.py     # Scrapes and saves game ID lists by season
│   └── collect_games.py        # Downloads raw game JSON files in parallel
├── tests/                      # Unit tests
│   ├── test_parsers.py         # Tests for ESPN JSON parsing logic
│   ├── test_features.py        # Tests for feature engineering transformations
│   └── test_calibration.py     # Tests for calibration metric calculations
├── data/                       # Raw and processed data (not tracked in git)
│   ├── raw/
│   │   ├── games/              # Gzipped ESPN game summary JSON files
│   │   ├── schedules/          # Per-season game ID lists from ESPN scoreboard
│   │   └── lines/              # Per-season Vegas lines JSON from CFBD
│   └── processed/
│       ├── plays.parquet       # Play-level dataset with win probabilities + features
│       └── games.parquet       # Game-level dataset with metadata and spread info
├── blog_post_draft.md          # Draft write-up of findings
├── pyproject.toml              # Project dependencies (managed with uv)
└── .env.example                # Template for required environment variables
```

---

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
uv sync
```

Copy `.env.example` to `.env` and add your CFBD API key if you want to scrape the CFPB data for betting odds (data/raw/lines/):

```bash
cp .env.example .env
```

---

## Running the Analysis

Run the notebooks in order, or use the scripts for data collection:

```bash
# Collect game IDs for seasons 2021–2023
python scripts/collect_schedule.py

# Download raw game summaries (play-by-play + win probability)
python scripts/collect_games.py
```

Then build the processed Parquet files:

```python
from cfb_calibration.processing.pipeline import build_dataset
from pathlib import Path

build_dataset(
    raw_games_dir=Path("data/raw/games"),
    schedules_dir=Path("data/raw/schedules"),
    lines_dir=Path("data/raw/lines"),
    output_dir=Path("data/processed"),
    seasons=[2021, 2022, 2023],
)
```

Open `notebooks/03_calibration_analysis.ipynb` for the full analysis and figures.

---

## Running Tests

```bash
uv run pytest
```
