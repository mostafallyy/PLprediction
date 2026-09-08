"""
Phase 2e - Player-season stats + impact tiers for the PL Match Predictor.

Cleans three raw copy-pasted FBref/Sports Reference table dumps -
PLAYERDATA.txt ("Player Standard Stats"), dfDAtA.txt ("Player
Miscellaneous Stats" - fouls, cards, tackles won, interceptions), and
GoalKeeper.txt ("Player Goalkeeping" - saves, goals against, clean
sheets) - into one tidy player-season CSV, then classifies every
player-season into one of 5 impact tiers within their own squad that
season. All three cover 2015-16 through 2026-27 (12 seasons - as far
back as FBref's free tables go).

Per FBref's stated data-usage terms ("When using SR data, please cite
us and provide a link and/or a mention") this project credits FBref -
see the README data-sources section.

Why season-level, not match-level: FBref's free tables are season
aggregates (goals/tackles/saves accumulated so far that season), not
per-match box scores. Using THIS season's running totals as a
mid-season feature would leak the season's own future results into
early-season predictions. So gold_features.py does NOT use a team's
current-season aggregate - it uses the PRIOR COMPLETED season's squad
totals as a fixed "squad strength carried over" feature for every
match in the following season. That's leakage-safe: nothing about a
match's own season is used to predict it.

Joining the three tables: all three carry the same FBref player-id
hash (verified against known players - e.g. Mohamed Salah is
"e342ad68" in every one of them). Joined on
(player_id, season_start_year, squad) rather than just
(player_id, season_start_year), because a player transferred mid-season
gets a separate row per club in EVERY one of these tables (e.g. Danny
Ings, Aston Villa -> West Ham in 2022-23) - joining without the squad
would silently cross-multiply those rows.

Raw file quirks handled here:
  - PLAYERDATA.txt: 14 season blocks pasted back to back; 2 are exact
    duplicates (2025-26, 2016-17 each pasted twice) - dropped.
  - FBref's Age column is "25" in some exports and "29-335"
    (years-days) in others, depending on when the snapshot was pulled -
    normalized to a plain float (years + days/365).
  - Squad names ("Manchester Utd", a truncated "Nottingham") didn't
    all match the alias table in team_names.py - extended it rather
    than silently losing those two clubs' rows. (Checked: dfDAtA.txt
    and GoalKeeper.txt's squad names all matched cleanly already.)
  - Season-block titles use different word positions across tables
    ("Player Standard Stats 2016-2017 Premier League" vs "Player
    Goalkeeping 2020-2021 Premier League (per 90)") - parsed with a
    regex on the four-digit-dash-four-digit pattern instead of a fixed
    word index, so a stray suffix like "(per 90)" can't break it.

Impact classification (5 tiers, computed WITHIN each squad-season,
across ALL players in that squad together - outfielders and the
goalkeeper(s) on one unified ranking, never across squads - a "Tier 1"
at a relegation candidate and a "Tier 1" at a title contender aren't
claimed to be equally good players, just equally central to THEIR OWN
team that season):

  impact_score = minutes_share * (1 + 0.5 * contribution)

  minutes_share = this player's 90s played / the SQUAD's total 90s
  played that season (across every player, GKs included) - the
  dominant term on purpose: minutes trusted by the manager is the best
  available proxy for "how central is this player to this team."

  contribution is position-aware, because a defender who never scores
  isn't low-impact, and a goalkeeper's job has nothing to do with G+A:
    - Outfield: contribution = attack_weight(pos) * attack_per90
                              + defense_weight(pos) * defense_per90
      attack_per90  = non-penalty (goals + assists) per 90 (gapk_per90)
      defense_per90 = (tackles won + interceptions) per 90, from the
                       Miscellaneous Stats table
      weights lean toward attack for forwards, defense for defenders,
      and split evenly for midfielders (see ATTACK_WEIGHT below) - a
      hybrid tag like "FWMF" averages the weights of both codes it
      contains.
    - Goalkeepers: contribution = gk_quality, built from save
      percentage and clean-sheet percentage relative to a league-ish
      baseline (65% saves, 25% clean sheets) - the only shot-stopping
      signal this free table exposes; there's no shot-quality-adjusted
      (PSxG) data available, so this is a rough proxy, not a proper
      goalkeeping model. GK rows fall out of ATTACK_WEIGHT entirely and
      use this instead.

  Players are then ranked by impact_score within (squad_key, season)
  and split into 5 equal-count tiers by rank (not qcut - qcut breaks
  on the many tied/near-zero scores from single-appearance players).

Tiers: 1 = Talisman (top ~20% by impact), 2 = Key Player,
3 = Regular Starter, 4 = Squad Rotation, 5 = Fringe/Backup.

Output: external_data/player_season_stats.csv, tracked in git (same
reasoning as the Kaggle CSV and manager tenures - the deployed service
needs to see it too).
"""

import csv
import math
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from team_names import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STANDARD_RAW_PATH = ROOT / "PLAYERDATA.txt"
MISC_RAW_PATH = ROOT / "dfDAtA.txt"
GK_RAW_PATH = ROOT / "GoalKeeper.txt"
OUT_PATH = ROOT / "external_data" / "player_season_stats.csv"

TIER_LABELS = {
    1: "Talisman",
    2: "Key Player",
    3: "Regular Starter",
    4: "Squad Rotation",
    5: "Fringe/Backup",
}

# How much an outfield player's IMPACT depends on attacking output
# (non-penalty G+A/90) vs. defensive output (tackles won + interceptions
# per 90). Pure GK is handled separately via gk_quality, not this table.
ATTACK_WEIGHT = {"FW": 1.0, "MF": 0.5, "DF": 0.15}
DEFENSE_WEIGHT = {"FW": 0.15, "MF": 0.5, "DF": 1.0}

# Column layout after FBref's two-row header (category row + name row)
# collapses to one row - verified against the raw files by hand.
STANDARD_COLUMNS = [
    "rk", "player", "nation", "pos", "squad", "age_raw", "born",
    "mp", "starts", "min", "nineties",
    "gls", "ast", "g_plus_a", "gpk", "pk", "pkatt", "crdy", "crdr",
    "gls_per90", "ast_per90", "g_plus_a_per90", "gpk_per90", "gapk_per90",
    "matches_link", "player_id",
]
MISC_COLUMNS = [
    "rk", "player", "nation", "pos", "squad", "age_raw", "born", "nineties",
    "crdy", "crdr", "crdy2", "fls", "fld", "off", "crs", "int_", "tklw",
    "pkwon", "pkcon", "og", "matches_link", "player_id",
]
GK_COLUMNS = [
    "rk", "player", "nation", "pos", "squad", "age_raw", "born",
    "mp", "starts", "min", "nineties",
    "ga", "ga90", "sota", "saves", "save_pct", "w", "d", "l", "cs", "cs_pct",
    "pkatt", "pka", "pksv", "pkm", "save_pct2", "matches_link", "player_id",
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


def parse_season_blocks(lines: list[str], block_prefix: str, columns: list[str]) -> pd.DataFrame:
    """Generic parser for a "<block_prefix> <season> Premier League[...]"
    -> "Rk,Player,..." -> data-rows dump, shared by all three raw files.
    The season year is pulled with a regex (not a fixed word index),
    since GoalKeeper.txt's titles carry an extra "(per 90)" suffix on
    one block that shifts word positions.
    """
    starts = [i for i, l in enumerate(lines) if l.startswith(block_prefix)]
    starts.append(len(lines))

    rows = []
    for i in range(len(starts) - 1):
        block = lines[starts[i]:starts[i + 1]]
        title = block[0].strip()
        m = re.search(r"(\d{4})-\d{4}", title)
        if not m:
            continue
        season_start_year = int(m.group(1))

        hdr_idx = next((j for j, l in enumerate(block) if l.startswith("Rk,Player")), None)
        if hdr_idx is None:
            continue

        for line in block[hdr_idx + 1:]:
            if not line.strip() or not line[0].isdigit():
                continue
            parsed = next(csv.reader([line]))
            if len(parsed) != len(columns):
                continue  # malformed row - skip rather than guess
            row = dict(zip(columns, parsed))
            row["season_start_year"] = season_start_year
            rows.append(row)

    return pd.DataFrame(rows)


def attack_weight(pos: str) -> float:
    codes = [c for c in ATTACK_WEIGHT if c in str(pos)]
    if not codes:
        return 0.5  # unknown position code - neutral
    return sum(ATTACK_WEIGHT[c] for c in codes) / len(codes)


def defense_weight(pos: str) -> float:
    codes = [c for c in DEFENSE_WEIGHT if c in str(pos)]
    if not codes:
        return 0.5
    return sum(DEFENSE_WEIGHT[c] for c in codes) / len(codes)


# Small-sample shrinkage: a player with 2-3 nineties who happens to log a
# hot game (say, 9 tackles+interceptions in one sub appearance) produces an
# extreme per-90 RATE purely from sample-size noise, and without correction
# that one hot game can outrank a nailed-on starter's full season. Shrink
# every player's per-90 rate toward the season-wide (nineties-weighted)
# average rate, with PRIOR_NINETIES worth of "pseudo-minutes" of that
# average mixed in - standard empirical-Bayes treatment for small-sample
# rate stats. A player with a full season of minutes is barely affected;
# a two-appearance player is pulled hard toward the league-typical rate.
PRIOR_NINETIES = 4.0


def _shrink_per90(totals: pd.Series, nineties: pd.Series, prior_rate: float) -> pd.Series:
    return (totals + prior_rate * PRIOR_NINETIES) / (nineties + PRIOR_NINETIES)


def classify_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Adds impact_score, team_rank, tier (1-5), tier_label - computed
    within each (squad_key, season_start_year) group, across ALL
    players (outfield + GK) in that squad-season together."""
    df = df.copy()
    squad_total_nineties = df.groupby(["squad_key", "season_start_year"])["nineties"].transform("sum")
    minutes_share = (df["nineties"] / squad_total_nineties).fillna(0)

    is_gk = df["pos"].astype(str).str.startswith("GK")

    # --- outfield contribution: position-weighted attack + defense,
    # shrunk toward each SEASON's typical rate (computed from players with
    # a real sample - >= PRIOR_NINETIES - so the prior itself isn't
    # contaminated by the same small-sample noise it's meant to correct) ---
    attack_total = df["gapk_per90"].fillna(0) * df["nineties"].fillna(0)
    defense_total = df["tklw"].fillna(0) + df["int_"].fillna(0)

    reliable = df["nineties"] >= PRIOR_NINETIES
    season_attack_prior = (
        attack_total.where(reliable).groupby(df["season_start_year"]).transform("sum")
        / df["nineties"].where(reliable).groupby(df["season_start_year"]).transform("sum")
    ).fillna(attack_total.sum() / max(df["nineties"].sum(), 1))
    season_defense_prior = (
        defense_total.where(reliable).groupby(df["season_start_year"]).transform("sum")
        / df["nineties"].where(reliable).groupby(df["season_start_year"]).transform("sum")
    ).fillna(defense_total.sum() / max(df["nineties"].sum(), 1))

    attack_per90 = (attack_total + season_attack_prior * PRIOR_NINETIES) / (df["nineties"].fillna(0) + PRIOR_NINETIES)
    defense_per90 = (defense_total + season_defense_prior * PRIOR_NINETIES) / (df["nineties"].fillna(0) + PRIOR_NINETIES)
    aw = df["pos"].apply(attack_weight)
    dw = df["pos"].apply(defense_weight)
    outfield_contribution = aw * attack_per90 + dw * defense_per90

    # --- goalkeeper contribution: save%/clean-sheet% vs. a rough baseline,
    # shrunk toward the season average the same way (a keeper's one cameo
    # appearance with a clean sheet shouldn't outrank a full-season starter
    # any more than an outfield player's one hot game should) ---
    save_pct_raw = pd.to_numeric(df["save_pct"], errors="coerce")
    cs_pct_raw = pd.to_numeric(df["cs_pct"], errors="coerce")
    gk_nineties = df["nineties"].where(is_gk)
    gk_reliable = is_gk & (df["nineties"] >= PRIOR_NINETIES)
    season_save_prior = (
        (save_pct_raw.where(gk_reliable) * gk_nineties.where(gk_reliable)).groupby(df["season_start_year"]).transform("sum")
        / gk_nineties.where(gk_reliable).groupby(df["season_start_year"]).transform("sum")
    ).fillna(65)
    season_cs_prior = (
        (cs_pct_raw.where(gk_reliable) * gk_nineties.where(gk_reliable)).groupby(df["season_start_year"]).transform("sum")
        / gk_nineties.where(gk_reliable).groupby(df["season_start_year"]).transform("sum")
    ).fillna(25)
    save_pct = (save_pct_raw.fillna(season_save_prior) * gk_nineties.fillna(0) + season_save_prior * PRIOR_NINETIES) / (gk_nineties.fillna(0) + PRIOR_NINETIES)
    cs_pct = (cs_pct_raw.fillna(season_cs_prior) * gk_nineties.fillna(0) + season_cs_prior * PRIOR_NINETIES) / (gk_nineties.fillna(0) + PRIOR_NINETIES)
    gk_quality = ((save_pct - 65) / 100) + ((cs_pct - 25) / 100)

    contribution = outfield_contribution.where(~is_gk, gk_quality)
    df["impact_score"] = (minutes_share * (1 + 0.5 * contribution)).round(4)

    df["team_rank"] = df.groupby(["squad_key", "season_start_year"])["impact_score"] \
        .rank(method="first", ascending=False).astype(int)
    group_size = df.groupby(["squad_key", "season_start_year"])["impact_score"].transform("size")
    percentile = df["team_rank"] / group_size
    df["tier"] = percentile.apply(lambda p: min(5, math.ceil(p * 5)))
    df["tier_label"] = df["tier"].map(TIER_LABELS)
    return df


def main():
    if not STANDARD_RAW_PATH.exists():
        sys.exit(f"{STANDARD_RAW_PATH} not found.")
    if not MISC_RAW_PATH.exists():
        sys.exit(f"{MISC_RAW_PATH} not found.")
    if not GK_RAW_PATH.exists():
        sys.exit(f"{GK_RAW_PATH} not found.")

    std_lines = STANDARD_RAW_PATH.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    std = parse_season_blocks(std_lines, "Player Standard Stats", STANDARD_COLUMNS)
    n_std_raw = len(std)
    # exact-duplicate season blocks (2025-26, 2016-17 each pasted twice)
    std = std.drop_duplicates(subset=[c for c in std.columns if c != "matches_link"])
    n_std_deduped = n_std_raw - len(std)

    misc_lines = MISC_RAW_PATH.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    misc = parse_season_blocks(misc_lines, "Player Miscellaneous Stats", MISC_COLUMNS)
    misc = misc.drop_duplicates(subset=["player_id", "season_start_year", "squad"])

    gk_lines = GK_RAW_PATH.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    gk = parse_season_blocks(gk_lines, "Player Goalkeeping", GK_COLUMNS)
    gk = gk.drop_duplicates(subset=["player_id", "season_start_year", "squad"])

    std["age"] = std["age_raw"].apply(parse_age)
    std["squad_key"] = std["squad"].apply(normalize)

    std_numeric = ["mp", "starts", "min", "nineties", "gls", "ast", "g_plus_a", "gpk",
                    "pk", "pkatt", "crdy", "crdr", "gls_per90", "ast_per90",
                    "g_plus_a_per90", "gpk_per90", "gapk_per90"]
    for c in std_numeric:
        std[c] = pd.to_numeric(std[c], errors="coerce")

    misc_numeric = ["nineties", "crdy", "crdr", "crdy2", "fls", "fld", "off", "crs",
                     "int_", "tklw", "pkwon", "pkcon", "og"]
    for c in misc_numeric:
        misc[c] = pd.to_numeric(misc[c], errors="coerce")

    gk_numeric = ["mp", "starts", "min", "nineties", "ga", "ga90", "sota", "saves",
                  "save_pct", "w", "d", "l", "cs", "cs_pct", "pkatt", "pka", "pksv", "pkm"]
    for c in gk_numeric:
        gk[c] = pd.to_numeric(gk[c], errors="coerce")

    # left-join defensive + goalkeeping columns onto the standard-stats base,
    # keyed on (player_id, season, squad) so mid-season transfers don't
    # cross-multiply rows (every table splits a transferred player's season
    # into one row per club).
    misc_cols = ["player_id", "season_start_year", "squad", "fls", "fld", "off", "crs", "int_", "tklw", "pkwon", "pkcon", "og"]
    gk_cols = ["player_id", "season_start_year", "squad", "ga", "ga90", "sota", "saves", "save_pct", "cs", "cs_pct", "pksv"]

    df = std.merge(misc[misc_cols], on=["player_id", "season_start_year", "squad"], how="left")
    df = df.merge(gk[gk_cols], on=["player_id", "season_start_year", "squad"], how="left")

    df["def_actions_per90"] = ((df["tklw"].fillna(0) + df["int_"].fillna(0)) / df["nineties"].replace(0, float("nan"))).round(3)

    df = classify_impact(df)

    out = df[[
        "player", "player_id", "nation", "pos", "squad", "squad_key",
        "age", "born", "season_start_year",
        "mp", "starts", "min", "nineties",
        "gls", "ast", "g_plus_a", "gpk", "pk", "pkatt", "crdy", "crdr",
        "gls_per90", "ast_per90", "g_plus_a_per90", "gpk_per90", "gapk_per90",
        "tklw", "int_", "def_actions_per90", "fls", "fld", "off", "crs",
        "ga", "ga90", "sota", "saves", "save_pct", "cs", "cs_pct", "pksv",
        "impact_score", "team_rank", "tier", "tier_label",
    ]].rename(columns={"int_": "int"}).sort_values(["season_start_year", "squad_key", "team_rank"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    n_with_defense = out["tklw"].notna().sum()
    n_gk_rows = (out["pos"].astype(str).str.startswith("GK")).sum()
    n_gk_with_saves = out.loc[out["pos"].astype(str).str.startswith("GK"), "saves"].notna().sum()

    print(f"Player-season stats: {len(out)} rows ({n_std_raw} raw standard-stats rows, "
          f"{n_std_deduped} exact-duplicate rows dropped) across {out['season_start_year'].nunique()} "
          f"seasons, {out['squad_key'].nunique()} squads -> {OUT_PATH}")
    print(f"Seasons: {sorted(out['season_start_year'].unique())}")
    print(f"Defensive-stats coverage (tackles won/interceptions matched from Misc table): "
          f"{n_with_defense}/{len(out)} rows ({n_with_defense / len(out):.1%})")
    print(f"Goalkeeping-stats coverage: {n_gk_with_saves}/{n_gk_rows} GK-position rows matched")
    print("Tier distribution:")
    print(out["tier_label"].value_counts().reindex(TIER_LABELS.values()).to_string())


if __name__ == "__main__":
    main()
