"""
Phase 1 - Bronze ingestion for the PL Match Predictor.

Pulls fixtures, results, standings, and team/squad data for the Premier
League from football-data.org and lands the RAW JSON, untouched, into a
date-partitioned bronze folder. No cleaning/joining here on purpose -
that's the silver layer's job.

Pulls TWO seasons of matches:
  - the current season (for live/upcoming fixtures to predict)
  - the most recently completed season(s) (so there's enough finished,
    feature-complete history to train a model on early in a new season)

The `/competitions/PL/teams` call also returns each team's crest URL and
full squad roster in one shot (no extra API calls needed) - used by the
dashboard for team badges and squad lists.

Usage:
    export FOOTBALL_DATA_API_KEY=xxxx
    python src/ingest_bronze.py
"""

import os
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

API_BASE = "https://api.football-data.org/v4"
COMPETITION = "PL"  # Premier League
BRONZE_DIR = Path(__file__).resolve().parent.parent / "data" / "bronze"

# football-data.org "season" is the year the season STARTS in (e.g. 2025 -> 2025-26)
CURRENT_SEASON_START_YEAR = 2026
TRAIN_SEASON_START_YEARS = [2025, 2024, 2023]  # last 3 completed seasons available on the free tier (2022 and earlier return 403)


def get_api_key() -> str:
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("FOOTBALL_DATA_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit(
            "Missing FOOTBALL_DATA_API_KEY. Get a free key at "
            "https://www.football-data.org/client/register and export it:\n"
            "  export FOOTBALL_DATA_API_KEY=your_key_here"
        )
    return key


def fetch(endpoint: str, api_key: str, params: dict | None = None) -> dict:
    url = f"{API_BASE}/{endpoint}"
    resp = requests.get(url, headers={"X-Auth-Token": api_key}, params=params or {}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def land(payload: dict, name: str, run_date: str) -> Path:
    out_dir = BRONZE_DIR / name / f"pull_date={run_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def main():
    api_key = get_api_key()
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print(f"[{run_date}] Pulling Premier League data from football-data.org ...")

    # 1) Matches - current season + recent completed seasons, merged into one payload
    all_seasons = [CURRENT_SEASON_START_YEAR] + TRAIN_SEASON_START_YEARS
    merged_matches = []
    seen_ids = set()
    for season in all_seasons:
        try:
            payload = fetch(f"competitions/{COMPETITION}/matches", api_key, params={"season": season})
        except requests.HTTPError as e:
            print(f"  ! Failed to pull matches for season {season}: {e}")
            continue
        season_matches = payload.get("matches", [])
        new = 0
        for m in season_matches:
            if m["id"] not in seen_ids:
                m["_season_start_year"] = season
                merged_matches.append(m)
                seen_ids.add(m["id"])
                new += 1
        print(f"  - matches (season {season}-{season+1}): {new} records")

    out_path = land({"matches": merged_matches}, "matches", run_date)
    print(f"  -> landed {len(merged_matches)} total matches -> {out_path}")

    # 2) Standings + teams (crests + full squads in one call) - current season
    for name, endpoint in {
        "standings": f"competitions/{COMPETITION}/standings",
        "teams": f"competitions/{COMPETITION}/teams",
    }.items():
        try:
            payload = fetch(endpoint, api_key)
        except requests.HTTPError as e:
            print(f"  ! Failed to pull {name}: {e}")
            continue
        out_path = land(payload, name, run_date)
        print(f"  - {name}: landed -> {out_path}")

    print("Bronze ingestion complete.")


if __name__ == "__main__":
    main()
