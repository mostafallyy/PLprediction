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

Rolling form and head-to-head are joined on `team_key` (a normalized
club-name string, see team_names.py), not the numeric team_id -
football-data.org (2023-24+) and the Kaggle historical CSV (2000/01
through 2022-23) don't share an id space, so team_key is what lets a
club's Kaggle-era matches and its football-data.org-era matches feed
the SAME rolling form calculation. team_id/crest are still carried
through as nullable columns purely for serving (crest images, the
/team and /h2h endpoints) - Kaggle rows simply don't have them.

Manager tenure (external_data/manager_tenures.csv, scraped from
Wikipedia) and squad prior-season stats (external_data/
player_season_stats.csv, cleaned from FBref) are joined the same
as-of/lookup way: real facts knowable strictly before kickoff.

CORE_FEATURES vs ENRICHMENT_FEATURES (used downstream by
src/train_model.py and src/app.py, imported from here so the split
can't drift out of sync between training and serving):
  - CORE: rolling form, home/away splits, rest days. Missing on these
    genuinely means "we know nothing about this team" (its first-ever
    match in our combined history) - rows missing these are dropped,
    not imputed, same as before.
  - ENRICHMENT: head-to-head, manager tenure, prior-season squad
    stats. These have much lower coverage by nature (h2h needs a prior
    meeting; manager tenure only ~98% covered; prior-season squad
    stats only exist for 2016-17 onward, since that's as far back as
    the FBref data goes - 0% coverage before that). Dropping rows
    missing ANY of these would have gutted the dataset (squad-stats
    coverage alone is only ~33% of all finished matches - the other
    67% are pre-2016 Kaggle seasons). Instead these are median-imputed
    using ONLY the training split's median (see train_model.py) -
    standard practice, and a team missing a prior-season squad stat is
    treated as "average" rather than thrown out entirely.

Deliberately NOT added: the Kaggle CSV's extra match-stat columns
(shots, corners, cards) and current-season player aggregates - both
would either go stale for live predictions or leak the season's own
future results into early-season matches. See ingest_player_stats.py's
docstring for the leakage argument on the squad-stats feature.

Features:
  - rolling form (points per game, goals for/against) over last N matches
  - home/away specific form splits
  - head-to-head record between the two teams
  - rest days since each team's previous match
  - days the home/away manager has been in charge as of kickoff
  - home/away squad's PRIOR completed season: total goals, average
    squad age, squad size (players used) - a fixed, leakage-safe proxy
    for squad strength/depth/rotation carried into the new season
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SILVER_DIR = ROOT / "data" / "silver"
GOLD_DIR = ROOT / "data" / "gold"
MANAGER_TENURES_PATH = ROOT / "external_data" / "manager_tenures.csv"
PLAYER_SEASON_STATS_PATH = ROOT / "external_data" / "player_season_stats.csv"

FORM_WINDOW = 5  # last N matches for rolling form

CORE_FEATURES = [
    "home_form_ppg",
    "home_form_gf",
    "home_form_ga",
    "away_form_ppg",
    "away_form_gf",
    "away_form_ga",
    "home_home_ppg",
    "away_away_ppg",
    "home_rest_days",
    "away_rest_days",
]

ENRICHMENT_FEATURES = [
    "h2h_home_win_rate",
    "home_manager_tenure_days",
    "away_manager_tenure_days",
    "home_prev_season_goals",
    "away_prev_season_goals",
    "home_prev_season_avg_age",
    "away_prev_season_avg_age",
    "home_prev_season_squad_size",
    "away_prev_season_squad_size",
]

PRE_MATCH_FEATURES = CORE_FEATURES + ENRICHMENT_FEATURES

# Leakage discipline: every feature we build is tagged here.
FEATURE_TAGS = {
    **{f: "pre_match" for f in PRE_MATCH_FEATURES},
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
            "home_team_key": "team_key",
            "away_team_key": "opp_key",
            "home_goals": "gf",
            "away_goals": "ga",
        }
    ).assign(is_home=True)
    away = df.rename(
        columns={
            "away_team_key": "team_key",
            "home_team_key": "opp_key",
            "away_goals": "gf",
            "home_goals": "ga",
        }
    ).assign(is_home=False)

    cols = ["match_id", "utc_date", "team_key", "opp_key", "gf", "ga", "is_home", "status"]
    log = pd.concat([home[cols], away[cols]], ignore_index=True)
    log = log.sort_values(["team_key", "utc_date"]).reset_index(drop=True)
    return log


def team_form_asof(log: pd.DataFrame) -> pd.DataFrame:
    """
    For every (team, match) row - finished or scheduled - attach that
    team's rolling form as of strictly before that match, computed only
    from the team's own FINISHED matches (which may span both data
    sources, since they're keyed by the same normalized team_key).
    """
    out_frames = []
    for team_key, grp in log.groupby("team_key"):
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


def load_manager_tenures():
    if not MANAGER_TENURES_PATH.exists():
        return None
    t = pd.read_csv(MANAGER_TENURES_PATH, parse_dates=["start_date", "end_date"])
    t["start_date"] = t["start_date"].dt.tz_localize("UTC")
    t["end_date"] = t["end_date"].dt.tz_localize("UTC")
    return t


def manager_tenure_asof(log: pd.DataFrame, tenures):
    """For every (team, match) row, how many days had that team's manager
    been in charge as of kickoff - real historical fact knowable at match
    time, so no leakage risk (only start_date is used in the calculation).
    NaN when the club/date isn't covered by the scraped tenure table (a
    caretaker spell with an unparseable Wikipedia date, mostly) - an
    ENRICHMENT feature, so this gets imputed downstream, not dropped."""
    if tenures is None or tenures.empty:
        log = log.copy()
        log["manager_tenure_days"] = pd.NA
        return log

    out_frames = []
    for team_key, grp in log.groupby("team_key"):
        grp = grp.sort_values("utc_date").reset_index(drop=True)
        t = tenures[tenures["team_key"] == team_key].sort_values("start_date")
        if t.empty:
            grp["manager_tenure_days"] = pd.NA
            out_frames.append(grp)
            continue
        merged = pd.merge_asof(
            grp, t[["start_date", "end_date"]],
            left_on="utc_date", right_on="start_date", direction="backward",
        )
        # merge_asof only guarantees the most recent start <= match date -
        # still need to confirm the match actually falls before that
        # manager's recorded end_date (else it's a coverage gap, e.g. an
        # interim spell we dropped for an unparseable date).
        valid = merged["start_date"].notna() & (merged["end_date"].isna() | (merged["utc_date"] <= merged["end_date"]))
        merged["manager_tenure_days"] = (merged["utc_date"] - merged["start_date"]).dt.days.where(valid)
        merged = merged.drop(columns=["start_date", "end_date"])
        out_frames.append(merged)

    return pd.concat(out_frames, ignore_index=True)


def load_squad_prev_season_stats():
    """Aggregate FBref player-season rows to one row per (squad, season),
    then shift the season forward by one - so a lookup for season Y
    returns squad Y-1's totals. That's the leakage-safe "carried over
    from last season" feature described in ingest_player_stats.py."""
    if not PLAYER_SEASON_STATS_PATH.exists():
        return None
    p = pd.read_csv(PLAYER_SEASON_STATS_PATH)
    agg = p.groupby(["squad_key", "season_start_year"]).agg(
        prev_season_goals=("gls", "sum"),
        prev_season_avg_age=("age", "mean"),
        prev_season_squad_size=("player", "nunique"),
    ).reset_index()
    agg["season_start_year"] = agg["season_start_year"] + 1  # now means "the season this applies TO"
    return agg


def head_to_head_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-match home-win rate in this fixture's history, using only past
    meetings - keyed on team_key so Kaggle-era meetings count too."""
    seen = {}
    rates = []
    for _, row in df.sort_values("utc_date").iterrows():
        key = tuple(sorted([row["home_team_key"], row["away_team_key"]]))
        history = seen.get(key, [])
        rate = (sum(1 for h in history if h == row["home_team_key"]) / len(history)) if history else None
        rates.append(rate)
        if row["status"] == "FINISHED" and row["winner"] == "HOME_TEAM":
            seen.setdefault(key, []).append(row["home_team_key"])
        elif row["status"] == "FINISHED" and row["winner"] == "AWAY_TEAM":
            seen.setdefault(key, []).append(row["away_team_key"])
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

    tenures = load_manager_tenures()
    mgr = manager_tenure_asof(team_log, tenures)

    squad_stats = load_squad_prev_season_stats()

    form_lookup = form.set_index(["match_id", "team_key"])[
        ["form_ppg", "form_gf", "form_ga", "rest_days", "home_ppg_split", "away_ppg_split"]
    ]
    mgr_lookup = mgr.set_index(["match_id", "team_key"])[["manager_tenure_days"]]

    matches = head_to_head_rate(matches)

    if squad_stats is not None:
        matches = matches.merge(
            squad_stats.rename(columns={
                "squad_key": "home_team_key",
                "prev_season_goals": "home_prev_season_goals",
                "prev_season_avg_age": "home_prev_season_avg_age",
                "prev_season_squad_size": "home_prev_season_squad_size",
            }),
            on=["home_team_key", "season_start_year"], how="left",
        ).merge(
            squad_stats.rename(columns={
                "squad_key": "away_team_key",
                "prev_season_goals": "away_prev_season_goals",
                "prev_season_avg_age": "away_prev_season_avg_age",
                "prev_season_squad_size": "away_prev_season_squad_size",
            }),
            on=["away_team_key", "season_start_year"], how="left",
        )
    else:
        for c in ["home_prev_season_goals", "home_prev_season_avg_age", "home_prev_season_squad_size",
                  "away_prev_season_goals", "away_prev_season_avg_age", "away_prev_season_squad_size"]:
            matches[c] = pd.NA

    feat_rows = []
    for _, m in matches.iterrows():
        try:
            hf = form_lookup.loc[(m["match_id"], m["home_team_key"])]
            af = form_lookup.loc[(m["match_id"], m["away_team_key"])]
            hm = mgr_lookup.loc[(m["match_id"], m["home_team_key"])]
            am = mgr_lookup.loc[(m["match_id"], m["away_team_key"])]
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
                "home_manager_tenure_days": hm["manager_tenure_days"],
                "away_manager_tenure_days": am["manager_tenure_days"],
                "home_prev_season_goals": m["home_prev_season_goals"],
                "away_prev_season_goals": m["away_prev_season_goals"],
                "home_prev_season_avg_age": m["home_prev_season_avg_age"],
                "away_prev_season_avg_age": m["away_prev_season_avg_age"],
                "home_prev_season_squad_size": m["home_prev_season_squad_size"],
                "away_prev_season_squad_size": m["away_prev_season_squad_size"],
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

    n_finished = (gold["status"] == "FINISHED").sum()
    n_upcoming = gold["status"].isin(["SCHEDULED", "TIMED"]).sum()
    print(f"Gold feature table: {len(gold)} rows ({n_finished} finished, {n_upcoming} upcoming) -> {out_path}")
    print(f"Core (required) features: {CORE_FEATURES}")
    print(f"Enrichment (imputed if missing) features: {ENRICHMENT_FEATURES}")
    finished = gold[gold["status"] == "FINISHED"]
    for c in ENRICHMENT_FEATURES:
        cov = finished[c].notna().mean()
        print(f"  coverage - {c}: {cov:.1%}")


if __name__ == "__main__":
    main()
