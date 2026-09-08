"""
Phase 2f - Manager tier classification (top / mid / low by PL titles).

Unlike the player-tier classifier, this is NOT a fitted model - it's a
straight rule-based bucketing, because the target variable (PL titles
won) is what defines the tiers in the first place, and a fitted
classifier needs a spread of examples across classes to learn
anything. Titles-won is heavily right-skewed across the 311 managers
in manager_tenures.csv: 298 of them have never won the league, 9 have
won exactly once, and only 4 (Ferguson, Guardiola, Wenger, Mourinho)
have won more than once. A percentile/tercile split on that
distribution would be meaningless - it would put zero-title managers
in different tiers from each other for no real reason, since two
managers who've both won nothing aren't distinguishable by this
metric. The natural break points in the data ARE the tier boundaries:

  top  - 2+ Premier League titles (the sustained-dynasty bracket)
  mid  - exactly 1 Premier League title (won it once)
  low  - 0 Premier League titles (everyone else who's managed in the PL)

Output: external_data/manager_tiers.csv, tracked in git alongside
manager_tenures.csv/manager_titles.csv. One row per manager who has
ever managed a PL club in our tenure data (all 311), columns:
manager_name, titles_won, tier, clubs_managed (semicolon-joined, from
manager_tenures.csv, not just the title-winning clubs).
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TENURES_PATH = ROOT / "external_data" / "manager_tenures.csv"
TITLES_PATH = ROOT / "external_data" / "manager_titles.csv"
OUT_PATH = ROOT / "external_data" / "manager_tiers.csv"


def tier_for(titles: int) -> str:
    if titles >= 2:
        return "top"
    if titles == 1:
        return "mid"
    return "low"


def main():
    if not TENURES_PATH.exists():
        sys.exit(f"{TENURES_PATH} not found - run ingest_managers.py first.")
    if not TITLES_PATH.exists():
        sys.exit(f"{TITLES_PATH} not found - run ingest_manager_titles.py first.")

    tenures = pd.read_csv(TENURES_PATH)
    tenures["manager_name"] = tenures["manager_name"].str.rstrip("†‡*")
    titles = pd.read_csv(TITLES_PATH)

    clubs_by_manager = (
        tenures.groupby("manager_name")["team_key"]
        .apply(lambda s: ";".join(sorted(set(s))))
        .reset_index()
        .rename(columns={"team_key": "clubs_managed"})
    )

    out = clubs_by_manager.merge(
        titles[["manager_name", "titles_won"]], on="manager_name", how="left"
    )
    out["titles_won"] = out["titles_won"].fillna(0).astype(int)
    out["tier"] = out["titles_won"].apply(tier_for)
    out = out.sort_values(["titles_won", "manager_name"], ascending=[False, True])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    counts = out["tier"].value_counts()
    print(f"Classified {len(out)} managers -> {OUT_PATH}")
    for tier in ["top", "mid", "low"]:
        n = counts.get(tier, 0)
        names = out[out["tier"] == tier]["manager_name"].tolist()
        preview = ", ".join(names[:6]) + (f", +{n - 6} more" if n > 6 else "")
        print(f"  {tier:>4} (>={2 if tier=='top' else 1 if tier=='mid' else 0} titles): {n} managers - {preview}")


if __name__ == "__main__":
    main()
