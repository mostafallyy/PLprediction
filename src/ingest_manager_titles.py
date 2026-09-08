"""
Phase 2e - Premier League title counts per manager.

Scrapes Wikipedia's "List of Premier League seasons" page for the
season-by-season Champions column (34 seasons, 1992-93 through
2025-26), then cross-references each title-winning season against
our own manager_tenures.csv (built by ingest_managers.py from a
DIFFERENT Wikipedia page - "List of Premier League managers") to
figure out which specific manager was in charge of the champion club
when that title was won.

Why not just scrape a "manager X won N Premier Leagues" number
directly off Wikipedia? There's no single clean table for that - each
manager's title count only exists implicitly, scattered across
their own infobox/prose. Joining season-champion -> team -> tenure
dates is the one approach that reuses data we've already validated
(manager_tenures.csv) and stays fully reproducible/scriptable rather
than needing per-manager page scraping or hand-entry.

Matching rule: for each title-winning season, take the club's PL
title-clinching date as an approximation - May 25 of the season's
second calendar year (top-flight seasons have finished by then in
every one of these 34 seasons) - and pick whichever of that club's
manager_tenures rows covers that date (start_date <= date <= end_date,
end_date blank/NaT meaning "still in charge"). If more than one row
matches (shouldn't happen for a clean tenure table) the most recent
start_date wins. If zero rows match, the season is logged and left
unattributed rather than guessed at.

Output: external_data/manager_titles.csv, tracked in git alongside
manager_tenures.csv for the same "the deployed service needs to see
it too" reason. Columns: manager_name, titles_won, title_seasons
(semicolon-joined "1992-93;1993-94" list), clubs (semicolon-joined).

Re-run any time to refresh (needs network access - Wikipedia isn't
reachable from every environment this project runs in).
"""

import re
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from team_names import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TENURES_PATH = ROOT / "external_data" / "manager_tenures.csv"
OUT_PATH = ROOT / "external_data" / "manager_titles.csv"

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_Premier_League_seasons"
HEADERS = {"User-Agent": "pl-match-predictor-student-project/1.0"}

# Strip title-count digits and footnote symbols off "Manchester United 13"
# / "Chelsea 3 †" / "Manchester City 7 #" -> "Manchester United", etc.
_TRAILING_JUNK = re.compile(r"\s*\d+\s*[†#§\[\]a-z]*\s*$")


def fetch_champions() -> pd.DataFrame:
    tables = pd.read_html(WIKI_URL, storage_options=HEADERS)
    target = None
    for t in tables:
        cols = [c[0] if isinstance(c, tuple) else c for c in t.columns]
        if "Season" in cols and any("Champions" in str(c) for c in cols):
            target = t
            break
    if target is None:
        sys.exit("Could not find the season/champions table on the Wikipedia page - its layout may have changed.")

    target.columns = [c[0] if isinstance(c, tuple) else c for c in target.columns]
    champ_col = [c for c in target.columns if "Champions" in str(c)][0]

    rows = []
    for _, r in target[["Season", champ_col]].iterrows():
        season_raw = str(r["Season"]).strip()
        champ_raw = str(r[champ_col]).strip()
        m = re.match(r"(\d{4})", season_raw)
        if not m or champ_raw in ("", "nan"):
            continue
        season_start_year = int(m.group(1))
        club = _TRAILING_JUNK.sub("", champ_raw).strip()
        rows.append({"season_start_year": season_start_year, "season": season_raw, "club": club})

    return pd.DataFrame(rows)


def attribute_manager(club_key: str, season_start_year: int, tenures: pd.DataFrame):
    clinch_date = pd.Timestamp(year=season_start_year + 1, month=5, day=25)
    candidates = tenures[tenures["team_key"] == club_key].copy()
    if candidates.empty:
        return None
    covering = candidates[
        (candidates["start_date"] <= clinch_date)
        & (candidates["end_date"].isna() | (candidates["end_date"] >= clinch_date))
    ]
    if covering.empty:
        return None
    return covering.sort_values("start_date").iloc[-1]["manager_name"]


def main():
    if not TENURES_PATH.exists():
        sys.exit(f"{TENURES_PATH} not found - run ingest_managers.py first.")

    tenures = pd.read_csv(TENURES_PATH, parse_dates=["start_date", "end_date"])
    # manager_tenures.csv keeps Wikipedia footnote marks (dagger etc) that
    # ingest_managers.py's rstrip("‡*") doesn't cover - strip them here
    # rather than re-scraping, so title counts attach to a clean name.
    tenures["manager_name"] = tenures["manager_name"].str.rstrip("†‡*")
    champions = fetch_champions()
    if champions.empty:
        sys.exit("Parsed 0 champion rows - aborting rather than writing an empty/broken file.")

    champions["team_key"] = champions["club"].apply(normalize)

    unattributed = []
    records = []
    for _, row in champions.iterrows():
        mgr = attribute_manager(row["team_key"], row["season_start_year"], tenures)
        if mgr is None:
            unattributed.append(f"{row['season']} ({row['club']})")
            continue
        records.append({"manager_name": mgr, "season": row["season"], "club": row["club"]})

    if not records:
        sys.exit("Attributed 0 titles to any manager - join is broken, aborting.")

    titles_df = pd.DataFrame(records)
    grouped = (
        titles_df.groupby("manager_name")
        .agg(
            titles_won=("season", "count"),
            title_seasons=("season", lambda s: ";".join(sorted(s))),
            clubs=("club", lambda s: ";".join(sorted(set(s)))),
        )
        .reset_index()
        .sort_values("titles_won", ascending=False)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(OUT_PATH, index=False)

    print(f"Attributed {len(records)}/{len(champions)} title-seasons to {grouped['manager_name'].nunique()} managers -> {OUT_PATH}")
    print(grouped.to_string(index=False))
    if unattributed:
        print(f"\n  {len(unattributed)} season(s) could not be attributed (no covering tenure row):")
        for u in unattributed:
            print(f"    - {u}")


if __name__ == "__main__":
    main()
