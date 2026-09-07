"""
Phase 2a - Silver layer for the PL Match Predictor.

Reads the most recent bronze `matches` pull, flattens it into one row per
match with clean, typed columns, and writes it out as a silver Parquet
table. This is a straight clean/join step - no feature engineering here
(that's gold's job), just making the raw API shape usable.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BRONZE_DIR = ROOT / "data" / "bronze"
SILVER_DIR = ROOT / "data" / "silver"


def latest_pull_dir(name: str) -> Path:
    candidates = sorted((BRONZE_DIR / name).glob("pull_date=*"))
    if not candidates:
        sys.exit(f"No bronze data found for '{name}'. Run src/ingest_bronze.py first.")
    return candidates[-1]


def load_matches() -> dict:
    pull_dir = latest_pull_dir("matches")
    with open(pull_dir / "matches.json") as f:
        return json.load(f)


def flatten_matches(raw: dict) -> pd.DataFrame:
    rows = []
    for m in raw.get("matches", []):
        score = m.get("score", {}).get("fullTime", {})
        rows.append(
            {
                "match_id": m.get("id"),
                "utc_date": m.get("utcDate"),
                "status": m.get("status"),
                "matchday": m.get("matchday"),
                "home_team_id": m.get("homeTeam", {}).get("id"),
                "home_team_name": m.get("homeTeam", {}).get("name"),
                "away_team_id": m.get("awayTeam", {}).get("id"),
                "away_team_name": m.get("awayTeam", {}).get("name"),
                "home_goals": score.get("home"),
                "away_goals": score.get("away"),
                "winner": m.get("score", {}).get("winner"),  # HOME_TEAM / AWAY_TEAM / DRAW / None
            }
        )
    df = pd.DataFrame(rows)
    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True)
    df = df.sort_values("utc_date").reset_index(drop=True)
    return df


def main():
    raw = load_matches()
    df = flatten_matches(raw)

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SILVER_DIR / "matches.parquet"
    df.to_parquet(out_path, index=False)

    n_finished = (df["status"] == "FINISHED").sum()
    print(f"Silver matches table: {len(df)} rows ({n_finished} finished) -> {out_path}")


if __name__ == "__main__":
    main()
