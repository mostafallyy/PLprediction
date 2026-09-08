"""
Phase 2a - Silver layer for the PL Match Predictor.

Reads the most recent bronze `matches` pull (football-data.org, seasons
2023-24 onward - the free tier's floor) AND the Kaggle historical CSV
(data/bronze/kaggle_history/epl_final.csv, seasons 2000/01-2024/25),
flattens/unions them into one row-per-match silver table, and writes it
out as Parquet. This is a clean/join step - no feature engineering here
(that's gold's job).

Two sources, one schema:
  - football-data.org rows carry a real numeric team_id + crest URL and
    cover 2023-24 onward - this is what's used for live serving (crests,
    the /team and /h2h endpoints key off these ids).
  - Kaggle rows have no team id, only club names, and are used ONLY for
    seasons the API can't reach (season_start_year < 2023) - deeper
    rolling-form history for the model, never served directly.

Because the two sources don't share an id space, every row also gets a
`team_key` (see team_names.py) - a normalized club-name string that DOES
line up across both sources. gold_features.py joins/aggregates on
team_key instead of team_id so a club's Kaggle-era history and its
football-data.org-era history feed the same rolling form calculation.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from team_names import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BRONZE_DIR = ROOT / "data" / "bronze"
SILVER_DIR = ROOT / "data" / "silver"
# Tracked in git (unlike everything under data/, which is gitignored
# pipeline output) - this is a static external input, checked in on
# purpose so Render/Databricks deploys see the same historical depth
# as local dev, not just the last football-data.org pull.
KAGGLE_CSV = ROOT / "external_data" / "kaggle_epl_history.csv"

# The API's free tier only reaches this season and later - Kaggle fills
# in everything strictly before it so there's no double-counted overlap.
API_FLOOR_SEASON = 2023

_RESULT_TO_WINNER = {"H": "HOME_TEAM", "A": "AWAY_TEAM", "D": "DRAW"}


def latest_pull_dir(name: str) -> Path:
    candidates = sorted((BRONZE_DIR / name).glob("pull_date=*"))
    if not candidates:
        sys.exit(f"No bronze data found for '{name}'. Run src/ingest_bronze.py first.")
    return candidates[-1]


def load_matches() -> dict:
    pull_dir = latest_pull_dir("matches")
    with open(pull_dir / "matches.json") as f:
        return json.load(f)


def flatten_api_matches(raw: dict) -> pd.DataFrame:
    rows = []
    for m in raw.get("matches", []):
        score = m.get("score", {}).get("fullTime", {})
        home_name = m.get("homeTeam", {}).get("name")
        away_name = m.get("awayTeam", {}).get("name")
        rows.append(
            {
                "match_id": m.get("id"),
                "utc_date": m.get("utcDate"),
                "status": m.get("status"),
                "matchday": m.get("matchday"),
                "home_team_id": m.get("homeTeam", {}).get("id"),
                "home_team_name": home_name,
                "home_team_key": normalize(home_name),
                "home_team_crest": m.get("homeTeam", {}).get("crest"),
                "away_team_id": m.get("awayTeam", {}).get("id"),
                "away_team_name": away_name,
                "away_team_key": normalize(away_name),
                "away_team_crest": m.get("awayTeam", {}).get("crest"),
                "home_goals": score.get("home"),
                "away_goals": score.get("away"),
                "winner": m.get("score", {}).get("winner"),  # HOME_TEAM / AWAY_TEAM / DRAW / None
                "season_start_year": m.get("_season_start_year"),
                "source": "api",
            }
        )
    return pd.DataFrame(rows)


def flatten_kaggle_matches() -> pd.DataFrame:
    if not KAGGLE_CSV.exists():
        return pd.DataFrame()

    kg = pd.read_csv(KAGGLE_CSV)
    kg["season_start_year"] = kg["Season"].str.slice(0, 4).astype(int)
    kg = kg[kg["season_start_year"] < API_FLOOR_SEASON]
    if kg.empty:
        return pd.DataFrame()

    kg["match_date"] = pd.to_datetime(kg["MatchDate"], utc=True)
    # Deterministic synthetic id (stable across re-runs, unlike Python's
    # randomized str hash) - negative so it can never collide with a real
    # football-data.org match id, which is always a positive integer.
    import hashlib

    def synth_id(row) -> int:
        raw = f"{row['MatchDate']}|{row['HomeTeam']}|{row['AwayTeam']}"
        digest = hashlib.md5(raw.encode()).hexdigest()[:8]
        return -int(digest, 16)

    rows = []
    for _, r in kg.iterrows():
        rows.append(
            {
                "match_id": synth_id(r),
                "utc_date": r["match_date"],
                "status": "FINISHED",
                "matchday": None,
                "home_team_id": None,
                "home_team_name": r["HomeTeam"],
                "home_team_key": normalize(r["HomeTeam"]),
                "home_team_crest": None,
                "away_team_id": None,
                "away_team_name": r["AwayTeam"],
                "away_team_key": normalize(r["AwayTeam"]),
                "away_team_crest": None,
                "home_goals": int(r["FullTimeHomeGoals"]),
                "away_goals": int(r["FullTimeAwayGoals"]),
                "winner": _RESULT_TO_WINNER.get(r["FullTimeResult"]),
                "season_start_year": int(r["season_start_year"]),
                "source": "kaggle",
            }
        )
    return pd.DataFrame(rows)


def main():
    api_df = flatten_api_matches(load_matches())
    kaggle_df = flatten_kaggle_matches()

    df = pd.concat([api_df, kaggle_df], ignore_index=True)
    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True)
    df = df.sort_values("utc_date").reset_index(drop=True)

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SILVER_DIR / "matches.parquet"
    df.to_parquet(out_path, index=False)

    n_finished = (df["status"] == "FINISHED").sum()
    n_api = (df["source"] == "api").sum()
    n_kaggle = (df["source"] == "kaggle").sum()
    print(f"Silver matches table: {len(df)} rows ({n_finished} finished) -> {out_path}")
    print(f"  -> {n_api} from football-data.org (2023-24+), {n_kaggle} from Kaggle history (pre-2023-24)")


if __name__ == "__main__":
    main()
