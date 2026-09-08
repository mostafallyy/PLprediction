"""
Phase 2d - Manager tenure data for the PL Match Predictor.

Scrapes Wikipedia's "List of Premier League managers" page - a single,
freely-licensed (CC BY-SA) table covering every PL managerial
appointment since the league's founding in 1992, with From/Until dates
per club. This is what lets us compute a genuine pre-match feature -
"how long has this team's manager been in charge as of kickoff" - for
BOTH historical training rows and live upcoming fixtures, which is the
bar that matters: a feature that only exists for old matches and goes
stale for next week's fixtures isn't worth adding (see the project
plan's notes on why we did NOT add shot/corner rolling features from
the Kaggle CSV - those aren't available for 2023-24+ matches at all,
so they'd silently go stale for every live prediction).

Output: external_data/manager_tenures.csv, tracked in git for the same
reason as the Kaggle CSV - the deployed service needs to see it too,
not just local dev. Columns: team_key, manager_name, start_date,
end_date (nullable - blank means "still in charge as of this table's
last edit").

Re-run this script any time to refresh it (needs network access -
Wikipedia isn't reachable from every environment this project runs in,
which is exactly why the output is committed rather than fetched live
on every pipeline run).
"""

import sys
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from team_names import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "external_data" / "manager_tenures.csv"

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_Premier_League_managers"


def fetch_table() -> pd.DataFrame:
    resp = requests.get(WIKI_URL, headers={"User-Agent": "pl-match-predictor-student-project/1.0"}, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    tables = soup.find_all("table", {"class": "wikitable"})
    # The full appointment-history table is the first wikitable on the
    # page (columns: Name, Nat., Club, From, Until, Duration(days),
    # Years in League, Ref.) - verified by inspecting header row.
    target = None
    for t in tables:
        headers = [th.get_text(strip=True) for th in t.find_all("th")]
        if "From" in headers and "Until" in headers and "Club" in headers:
            target = t
            break
    if target is None:
        sys.exit("Could not find the manager appointment table on the Wikipedia page - its layout may have changed.")

    rows = []
    for tr in target.find_all("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if len(cells) < 5:
            continue
        name = cells[0].get_text(strip=True).rstrip("‡*")
        club = cells[2].get_text(strip=True)
        start = cells[3].get_text(strip=True)
        end = cells[4].get_text(strip=True)
        if not name or not club or not start:
            continue
        rows.append({"manager_name": name, "club": club, "start_raw": start, "end_raw": end})

    return pd.DataFrame(rows)


def main():
    df = fetch_table()
    if df.empty:
        sys.exit("Parsed 0 manager rows - aborting rather than writing an empty/broken file.")

    df["team_key"] = df["club"].apply(normalize)
    df["start_date"] = pd.to_datetime(df["start_raw"], format="%d %B %Y", errors="coerce")
    # "Until" is blank/"Present"/a future-looking placeholder for an
    # ongoing tenure on Wikipedia's convention - anything that doesn't
    # parse as a real past date is treated as still-ongoing (NaT/blank).
    df["end_date"] = pd.to_datetime(df["end_raw"], format="%d %B %Y", errors="coerce")

    n_unparsed_start = df["start_date"].isna().sum()
    df = df.dropna(subset=["start_date"])

    out = df[["team_key", "manager_name", "start_date", "end_date"]].sort_values(["team_key", "start_date"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"Manager tenure table: {len(out)} appointments across {out['team_key'].nunique()} clubs -> {OUT_PATH}")
    if n_unparsed_start:
        print(f"  ({n_unparsed_start} rows dropped - unparseable start date)")


if __name__ == "__main__":
    main()
