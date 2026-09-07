"""
Builds data/gold/teams.json: team_id -> name, crest, club colors, venue,
and squad roster (player name/position/nationality). Powers the
dashboard's team badges and squad panels.

Note: football-data.org's free tier does not expose actual starting-XI
lineups from past matches (that's a paid feature) - "squad" here is each
team's current registered roster, not a specific match's lineup.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRONZE_DIR = ROOT / "data" / "bronze"
GOLD_DIR = ROOT / "data" / "gold"


def latest_pull_dir(name: str) -> Path:
    candidates = sorted((BRONZE_DIR / name).glob("pull_date=*"))
    if not candidates:
        raise SystemExit(f"No bronze data found for '{name}'. Run src/ingest_bronze.py first.")
    return candidates[-1]


def main():
    teams_path = latest_pull_dir("teams") / "teams.json"
    with open(teams_path) as f:
        raw = json.load(f)

    teams = {}
    for t in raw.get("teams", []):
        squad = sorted(
            [
                {
                    "name": p.get("name"),
                    "position": p.get("position"),
                    "nationality": p.get("nationality"),
                }
                for p in t.get("squad", [])
            ],
            key=lambda p: (p["position"] or "", p["name"] or ""),
        )
        teams[str(t["id"])] = {
            "id": t["id"],
            "name": t.get("name"),
            "shortName": t.get("shortName"),
            "tla": t.get("tla"),
            "crest": t.get("crest"),
            "clubColors": t.get("clubColors"),
            "venue": t.get("venue"),
            "coach": (t.get("coach") or {}).get("name"),
            "squad": squad,
        }

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLD_DIR / "teams.json"
    out_path.write_text(json.dumps(teams, indent=2))
    print(f"Teams gold table: {len(teams)} teams -> {out_path}")


if __name__ == "__main__":
    main()
