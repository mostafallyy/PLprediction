"""
Phase 2b - Gold feature layer for the PL Match Predictor.

Builds one feature row per match - including SCHEDULED (upcoming)
fixtures - using ONLY information that would have been available before
kickoff (pre-match features). Every column is tagged in FEATURE_TAGS
below as "pre_match" (safe to train on / serve) or "post_match"
(leakage - label-only, never fed to the model).

For each team, we compute a rolling-form time series from their
FINISHED matches only, then "as-of" merge that onto every match
(finished or scheduled) using the most recent finished form strictly
before that match's date. This is what lets an upcoming, unplayed
fixture still get real pre-match features to predict on.

Features:
  - rolling form (points per game, goals for/against) over last N matches
  - home/away specific form splits
  - head-to-head record between the two teams
  - rest days since each team's previous match
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SILVER_DIR = ROOT / "data" / "silver"
GOLD_DIR = ROOT / "data" / "gold"

FORM_WINDOW = 5  # last N matches for rolling form

# Leakage discipline: every feature we build is tagged here.
FEATURE_TAGS = {
    "home_form_ppg": "pre_match",
    "home_form_gf": "pre_match",
    "home_form_ga": "pre_match",
    "away_form_ppg": "pre_match",
    "away_form_gf": "pre_match",
    "away_form_ga": "pre_match",
    "home_home_ppg": "pre_match",
    "away_away_ppg": "pre_match",
    "h2h_home_win_rate": "pre_match",
    "home_rest_days": "pre_match",
    "away_rest_days": "pre_match",
    # kept for reference / label construction only - NEVER train on these
    "home_goals": "post_match",
    "away_goals": "post_match",
    "winner": "post_match",
}


def points(goals_for, goals_against):
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def build_team_match_log(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, match) for ALL matches, finished or scheduled."""
    home = df.rename(
        columns={
            "home_team_id": "team_id",
            "away_team_id": "opp_id",
            "home_goals": "gf",
            "away_goals": "ga",
        }
    ).assign(is_home=True)
    away = df.rename(
        columns={
            "away_team_id": "team_id",
            "home_team_id": "opp_id",
            "away_goals": "gf",
            "home_goals": "ga",
        }
    ).assign(is_home=False)

    cols = ["match_id", "utc_date", "team_id", "opp_id", "gf", "ga", "is_home", "status"]
    log = pd.concat([home[cols], away[cols]], ignore_index=True)
    log = log.sort_values(["team_id", "utc_date"]).reset_index(drop=True)
    return log


def team_form_asof(log: pd.DataFrame) -> pd.DataFrame:
    """
    For every (team, match) row - finished or scheduled - attach that
    team's rolling form as of strictly before that match, computed only
    from the team's own FINISHED matches.
    """
    out_frames = []
    for team_id, grp in log.groupby("team_id"):
        grp = grp.sort_values("utc_date").reset_index(drop=True)

        finished = grp[grp["status"] == "FINISHED"].copy()
        finished["pts"] = finished.apply(lambda r: points(r["gf"], r["ga"]), axis=1)
        finished["form_ppg"] = finished["pts"].rolling(FORM_WINDOW, min_periods=1).mean()
        finished["form_gf"] = finished["gf"].rolling(FORM_WINDOW, min_periods=1).mean()
        finished["form_ga"] = finished["ga"].rolling(FORM_WINDOW, min_periods=1).mean()
        finished["home_pts"] = finished["pts"].where(finished["is_home"])
        finished["away_pts"] = finished["pts"].where(~finished["is_home"])
        finished["home_ppg_split"] = finished["home_pts"].expanding().mean()
        finished["away_ppg_split"] = finished["away_pts"].expanding().mean()
        finished["prev_match_date"] = finished["utc_date"]

        form_series = finished[
            ["utc_date", "form_ppg", "form_gf", "form_ga", "home_ppg_split",
             "away_ppg_split", "prev_match_date"]
        ].rename(columns={"utc_date": "asof_date"})

        # as-of merge: for each row in grp, pull the most recent finished-form
        # snapshot strictly BEFORE this match's date (avoids using same-day
        # leakage and, critically, never uses the match's own result).
        grp = grp.sort_values("utc_date")
        merged = pd.merge_asof(
            grp,
            form_series.sort_values("asof_date"),
            left_on="utc_date",
            right_on="asof_date",
            direction="backward",
            allow_exact_matches=False,
        )
        merged["rest_days"] = (merged["utc_date"] - merged["prev_match_date"]).dt.days
        out_frames.append(merged)

    return pd.concat(out_frames, ignore_index=True)


def head_to_head_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-match home-win rate in this fixture's history, using only past meetings."""
    seen = {}
    rates = []
    for _, row in df.sort_values("utc_date").iterrows():
        key = tuple(sorted([row["home_team_id"], row["away_team_id"]]))
        history = seen.get(key, [])
        rate = (sum(1 for h in history if h == row["home_team_id"]) / len(history)) if history else None
        rates.append(rate)
        if row["status"] == "FINISHED" and row["winner"] == "HOME_TEAM":
            seen.setdefault(key, []).append(row["home_team_id"])
        elif row["status"] == "FINISHED" and row["winner"] == "AWAY_TEAM":
            seen.setdefault(key, []).append(row["away_team_id"])
    df = df.sort_values("utc_date").copy()
    df["h2h_home_win_rate"] = rates
    return df


def main():
    silver_path = SILVER_DIR / "matches.parquet"
    if not silver_path.exists():
        raise SystemExit("No silver table found. Run src/silver_transform.py first.")

    matches = pd.read_parquet(silver_path)
    team_log = build_team_match_log(matches)
    form = team_form_asof(team_log)

    form_lookup = form.set_index(["match_id", "team_id"])[
        ["form_ppg", "form_gf", "form_ga", "rest_days", "home_ppg_split", "away_ppg_split"]
    ]

    matches = head_to_head_rate(matches)

    feat_rows = []
    for _, m in matches.iterrows():
        try:
            hf = form_lookup.loc[(m["match_id"], m["home_team_id"])]
            af = form_lookup.loc[(m["match_id"], m["away_team_id"])]
        except KeyError:
            continue
        feat_rows.append(
            {
                "match_id": m["match_id"],
                "utc_date": m["utc_date"],
                "matchday": m["matchday"],
                "home_team_id": m["home_team_id"],
                "home_team_name": m["home_team_name"],
                "home_team_crest": m["home_team_crest"],
                "away_team_id": m["away_team_id"],
                "away_team_name": m["away_team_name"],
                "away_team_crest": m["away_team_crest"],
                "status": m["status"],
                "season_start_year": m["season_start_year"],
                "home_form_ppg": hf["form_ppg"],
                "home_form_gf": hf["form_gf"],
                "home_form_ga": hf["form_ga"],
                "away_form_ppg": af["form_ppg"],
                "away_form_gf": af["form_gf"],
                "away_form_ga": af["form_ga"],
                "home_home_ppg": hf["home_ppg_split"],
                "away_away_ppg": af["away_ppg_split"],
                "home_rest_days": hf["rest_days"],
                "away_rest_days": af["rest_days"],
                "h2h_home_win_rate": m["h2h_home_win_rate"],
                # post-match, label-only columns - kept but flagged, never trained on
                "home_goals": m["home_goals"],
                "away_goals": m["away_goals"],
                "winner": m["winner"],
            }
        )

    gold = pd.DataFrame(feat_rows)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLD_DIR / "match_features.parquet"
    gold.to_parquet(out_path, index=False)

    pre_match_cols = [c for c, tag in FEATURE_TAGS.items() if tag == "pre_match"]
    n_finished = (gold["status"] == "FINISHED").sum()
    n_upcoming = gold["status"].isin(["SCHEDULED", "TIMED"]).sum()
    print(f"Gold feature table: {len(gold)} rows ({n_finished} finished, {n_upcoming} upcoming) -> {out_path}")
    print(f"Pre-match (safe) features: {pre_match_cols}")
    print(f"Post-match (label-only, never train on): {[c for c, t in FEATURE_TAGS.items() if t == 'post_match']}")


if __name__ == "__main__":
    main()
