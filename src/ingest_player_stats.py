"""
Phase 2e - Player-season stats for the PL Match Predictor.

Cleans PLAYERDATA.txt (raw copy-pasted "Player Standard Stats" tables
from FBref/Sports Reference, 2015-16 through 2026-27) into a tidy CSV.
Per FBref's stated data-usage terms ("When using SR data, please cite
us and provide a link and/or a mention") this project credits FBref -
see the README data-sources section.

Why season-level, not match-level: FBref's free tables are season
aggregates (goals/assists/cards accumulated so far that season), not
per-match box scores. Using THIS season's running totals as a
mid-season feature would leak the season's own future results into
early-season predictions (a team's May goal tally obviously "knows"
about matches that haven't happened yet in August). So gold_features.py
does NOT use a team's current-season aggregate - it uses the PRIOR
COMPLETED season's squad totals as a fixed "squad strength carried
over" feature for every match in the following season. That's
leakage-safe: nothing about a match's own season is used to predict it.

Raw file quirks handled here:
  - 14 season blocks pasted back to back; 2 are exact duplicates
    (2025-26, 2016-17 each pasted twice) - dropped.
  - FBref's Age column is "25" in some exports and "29-335"
    (years-days) in others, depending on when the snapshot was pulled -
    normalized to a plain float (years + days/365).
  - Squad names ("Manchester Utd", a truncated "Nottingham") didn't
    all match the alias table in team_names.py - extended it rather
    than silently losing those two clubs' rows.

Output: external_data/player_season_stats.csv, tracked in git (same
reasoning as the Kaggle CSV and manager tenures - the deployed service
needs to see it too).
"""

import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from team_names import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "PLAYERDATA.txt"
OUT_PATH = ROOT / "external_data" / "player_season_stats.csv"

# Column layout after FBref's two-row header (category row + name row)
# collapses to one row - verified against the raw file by hand.
COLUMNS = [
    "rk", "player", "nation", "pos", "squad", "age_raw", "born",
    "mp", "starts", "min", "nineties",
    "gls", "ast", "g_plus_a", "gpk", "pk", "pkatt", "crdy", "crdr",
    "gls_per90", "ast_per90", "g_plus_a_per90", "gpk_per90", "gapk_per90",
    "matches_link", "player_id",
]


def parse_age(raw: str):
    """'25' -> 25.0   '29-335' (29y 335d) -> 29.92"""
    if not raw:
        return None
    if "-" in raw:
        years, days = raw.split("-", 1)
        try:
            return round(int(years) + int(days) / 365, 2)
        except ValueError:
            return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_season_blocks(lines: list[str]) -> pd.DataFrame:
    starts = [i for i, l in enumerate(lines) if l.startswith("Player Standard Stats")]
    starts.append(len(lines))

    rows = []
    for i in range(len(starts) - 1):
        block = lines[starts[i]:starts[i + 1]]
        title = block[0].strip()
        # "Player Standard Stats 2016-2017 Premier League" -> 2016
        season_start_year = int(title.split()[3].split("-")[0])

        hdr_idx = next((j for j, l in enumerate(block) if l.startswith("Rk,Player")), None)
        if hdr_idx is None:
            continue

        for line in block[hdr_idx + 1:]:
            if not line.strip() or not line[0].isdigit():
                continue
            parsed = next(csv.reader([line]))
            if len(parsed) != len(COLUMNS):
                continue  # malformed row - skip rather than guess
            row = dict(zip(COLUMNS, parsed))
            row["season_start_year"] = season_start_year
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    if not RAW_PATH.exists():
        sys.exit(f"{RAW_PATH} not found.")

    lines = RAW_PATH.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    df = parse_season_blocks(lines)
    n_raw = len(df)

    # exact-duplicate season blocks (2025-26, 2016-17 each pasted twice)
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "matches_link"])
    n_deduped = n_raw - len(df)

    df["age"] = df["age_raw"].apply(parse_age)
    df["squad_key"] = df["squad"].apply(normalize)

    numeric_cols = ["mp", "starts", "min", "nineties", "gls", "ast", "g_plus_a", "gpk",
                     "pk", "pkatt", "crdy", "crdr", "gls_per90", "ast_per90",
                     "g_plus_a_per90", "gpk_per90", "gapk_per90"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = df[[
        "player", "player_id", "nation", "pos", "squad", "squad_key",
        "age", "born", "season_start_year",
        "mp", "starts", "min", "nineties",
        "gls", "ast", "g_plus_a", "gpk", "pk", "pkatt", "crdy", "crdr",
        "gls_per90", "ast_per90", "g_plus_a_per90", "gpk_per90", "gapk_per90",
    ]].sort_values(["season_start_year", "squad_key", "player"])


    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Player-season stats: {len(out)} rows ({n_raw} raw, {n_deduped} exact-duplicate rows dropped) "
          f"across {out['season_start_year'].nunique()} seasons, {out['squad_key'].nunique()} squads -> {OUT_PATH}")
    print(f"Seasons: {sorted(out['season_start_year'].unique())}")


if __name__ == "__main__":
    main()
