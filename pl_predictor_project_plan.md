# Premier League Match Predictor — Project Plan

A live-data pipeline that predicts Premier League match outcomes, built to demonstrate PySpark, ML pipelines, orchestration, and cloud deployment end to end.

## Why this project

Most student portfolios have a Jupyter notebook with a trained model. This has a live pipeline behind it — daily data ingestion, a medallion-architecture Spark transformation layer, a tracked and validated model, and a deployed prediction service. It's the same architecture pattern as your Kensington Tours internship work, but built on public data so it's fully shareable and provable.

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Data source | football-data.org (free tier) | Fixtures, results, standings for the Premier League |
| Processing | PySpark on Databricks Community Edition (free) | Real distributed Spark + Delta Lake, no cost |
| Storage | Delta Lake, bronze/silver/gold | Matches the medallion architecture you already know from Fabric |
| ML | LightGBM + MLflow | Same model family you shipped at Kensington; MLflow tracks experiments |
| Serving | FastAPI + Docker | Lightweight, standard, easy to demo |
| Hosting | Render (free tier) | No credit card, live public URL |
| Orchestration | Airflow (or GitHub Actions cron as a fallback) | Schedules the daily pull, retrain, and redeploy |
| Dashboard | Streamlit or Power BI | Power BI plays to your existing strength |

## Architecture

```
Football API → Bronze (raw JSON, Spark) → Silver (cleaned & joined)
                                              ↓
                          Gold layer & model training (LightGBM + MLflow)
                                              ↓
                          Prediction service (FastAPI, deployed on Render)
                                              ↓
                                  Dashboard (this week's predictions)

Whole pipeline runs daily via Airflow; model retrains weekly.
```

## Phase 1 — Data foundation

- [ ] Sign up for football-data.org free tier, get an API key
- [ ] Write a Python script pulling fixtures, results, and standings for the Premier League
- [ ] Land raw JSON untouched into a bronze Delta table on Databricks Community Edition
- [ ] Run it once manually and confirm the data lands correctly
- [ ] Set it up on a daily schedule (cron or Airflow)

**Done when:** you have a bronze table that updates itself every day with real match data.

## Phase 2 — Feature engineering & baseline model

- [ ] Build the silver layer in Spark: cleaned, joined match records
- [ ] Build the gold layer: feature table with recent form, home/away splits, head-to-head record, rest days between matches
- [ ] Tag every feature as pre-match (safe) or post-match (leakage) — the same discipline from your Kensington leakage audits
- [ ] Train a LightGBM baseline model with a **time-based** train/test split (not random — avoids leaking future form into the past)
- [ ] Track every run in MLflow (params, features used, AUC/accuracy)
- [ ] Record your validation numbers — you'll need these for the resume bullet later

**Done when:** you have a trained model with an honest, leakage-checked accuracy number you can defend in an interview.

## Phase 3 — Serving the predictions

- [ ] Build a FastAPI endpoint that loads the latest gold features and the trained model
- [ ] Endpoint returns win/draw/loss probabilities for upcoming fixtures
- [ ] Dockerize the service
- [ ] Deploy to Render's free tier — get a live, public URL

**Done when:** you can hit a URL and get back real predictions for this week's matches.

## Phase 4 — Orchestration & polish

- [ ] Wire ingestion → transform → retrain → deploy into one scheduled flow (Airflow, or GitHub Actions cron if Airflow setup drags)
- [ ] Build a lightweight dashboard (Streamlit or Power BI) showing this week's predictions
- [ ] Write the README as a short case study: problem, architecture, validation numbers, what you'd improve next
- [ ] Push the whole repo to GitHub, link it from your resume and portfolio

**Done when:** the whole thing runs itself daily with no manual steps, and there's a public writeup someone can read in two minutes.

## Timeline

Roughly 3 weeks at a couple hours per evening:

- **Week 1:** Phase 1 (get this right before touching modeling — a stale pipeline undermines the whole "live" selling point)
- **Week 2:** Phase 2
- **Week 3:** Phases 3 and 4

## Resume framing (once shipped)

> Built an end-to-end Premier League match prediction system: PySpark medallion pipeline (bronze/silver/gold) on Databricks, LightGBM model with leakage-checked, time-based validation (MLflow-tracked), served via a Dockerized FastAPI endpoint on a daily Airflow schedule.

This single project touches every keyword in your target Winter 2027 postings — PySpark, medallion architecture, MLflow, Docker, orchestration, cloud deployment — while being fun enough that you'll actually finish it.

## Stretch goals (only after Phase 4 is done)

- Add player-level data (injuries, suspensions) as features
- Expand to other leagues for a multi-league model
- Add a live in-match win-probability update using live score feeds
- A/B compare LightGBM against a simple Elo-rating baseline to quantify model lift

---

## Progress log (living notes — updated as we build)

**Status as of 2026-09-08:** Phases 1–3 shipped and live. Working through stretch/hardening items.

- [x] Phase 1 — Data foundation: football-data.org ingestion, bronze landing, `.env`-based API key.
- [x] Phase 2 — Silver/gold medallion transforms, leakage-tagged pre/post-match features (`merge_asof`-based, so upcoming fixtures get features too), time-based split, MLflow tracking.
- [x] Phase 3 — FastAPI serving (`/predictions`, `/matchweeks`, `/matchweek/{n}`, `/team/{id}`, `/h2h/{a}/{b}`), Dockerized, deployed live on Render: https://plprediction-met6.onrender.com (autoDeploy on push to `test` branch; pipeline re-runs and retrains on every cold start since `data/`/`models/` are gitignored).
- [x] Databricks (PySpark + Delta Lake, serverless Free Edition) mirror of the whole pipeline, run end-to-end via Jobs API (no classic clusters available on this workspace — serverless job compute used instead). Model published to a Delta table (`pl_predictor_gold.latest_model`, base64-encoded) since DBFS artifact download is blocked by Unity Catalog on this workspace. Daily 06:00 UTC schedule exists but is currently paused.
- [x] Custom "sexy" dark-theme frontend (not Streamlit/BI-style) — matchweek picker, animated probability bars, team crests, head-to-head modal, squad rosters. Deployed at the Render root URL.
- [x] Bug fix: matchdays were merging fixtures across all ingested seasons (30 fixtures showing under "matchday 1" instead of 10) — fixed by threading `season_start_year` through silver→gold and filtering matchweek views to the current season only (deliberately NOT applied to `/h2h`, which needs full history).
- [x] Bug fix: newly-promoted teams with no prior data showed a fabricated "-100% null" badge and fake 33/33/33 split instead of an honest "no prediction yet" state.
- [x] **2026-09-08: expanded training history + added a second model.** Bronze ingestion now pulls 4 seasons (2023–24 through 2026–27, 1520 matches total, 1170 finished) instead of 2 — football-data.org's free tier returns 403 for 2022 and earlier, so this is the max history available on this key. `train_model.py` now trains **both** LightGBM and Logistic Regression (StandardScaler + multinomial) on the identical time-based split and keeps whichever wins on held-out test log loss (a proper scoring rule, not just accuracy) — result this run: **Logistic Regression won** (log loss 1.044 vs LightGBM 1.068; naive baseline 1.090; accuracy 0.457 vs 0.426). `app.py` was updated to read `models/active_model.json` and dynamically load+call whichever model type is active, instead of hardcoding LightGBM.

### Open / next
- [ ] Considered pulling in the Kaggle "EPL match data 2000–2025" dataset (marcohuiii) for even deeper history — blocked for now: Kaggle's API needs a `kaggle.json` token (username + key) which isn't set up on this machine, and the dataset can't be pulled anonymously. Needs either the user's Kaggle API credentials, or a manual CSV download dropped into the repo, before it can be wired in as a second bronze source (would need its own team-name reconciliation against football-data.org's team IDs, since Kaggle datasets use their own naming).
- [ ] Databricks notebook `04_train_model.py` still only trains LightGBM and still uses the older 2-season ingestion — not yet updated to match the local pipeline's season expansion + dual-model comparison.
- [ ] Re-commit + push `test` branch (user pushes manually from PowerShell — device shell can't do interactive GitHub OAuth) and confirm Render redeploys cleanly with the new active model.
- [ ] Re-verify `/predictions` and the frontend after the Render redeploy picks up logistic regression as the active model.


### 2026-09-08 (later same day): wired in the Kaggle historical dataset

User supplied `epl_final.csv` (Kaggle: marcohuiii/english-premier-league-epl-match-data-2000-2025,
9380 matches, 2000/01–2024/25, with match stats but no team ids). Landed
it at `data/bronze/kaggle_history/epl_final.csv` as a second bronze
source and wired it into the pipeline properly rather than bolting it on:

- **The id problem:** football-data.org gives numeric team ids; Kaggle only has club names, and the two sources don't share an id space. Fix: added `src/team_names.py`, a name-normalization module (strips FC/AFC + punctuation, plus an alias table for ~15 clubs whose short name doesn't collapse to their full name — "Man City" → same key as "Manchester City FC", "Spurs" → "Tottenham Hotspur FC", etc.). Every row in silver now carries a `team_key` alongside the nullable `team_id`.
- **The overlap problem:** Kaggle runs through 2024/25, which overlaps the API's 2023-24+ coverage. Fix: Kaggle rows are only kept for `season_start_year < 2023` — API data wins for everything it covers (it has ids/crests/live status), Kaggle only fills the historical gap the free API tier can't reach (2000/01–2022/23).
- **gold_features.py** was re-keyed from `team_id` to `team_key` for the rolling-form and head-to-head joins, so a club's Kaggle-era matches and its football-data.org-era matches feed the *same* rolling-form calculation. `team_id`/crest are still carried through as nullable columns purely for serving (crest images, `/team`, `/h2h`) — Kaggle rows don't have them, which is fine since Kaggle rows are never served directly, only used to deepen training history.
- Verified match coverage: 26 of 27 current-era clubs matched cleanly to their Kaggle history; the one non-match (Luton Town) is correct, not a bug — their only PL season (2023-24) is already inside the API-covered range.
- **Result:** finished-match training set went from 1,170 → 9,820 rows (5,947 train / 1,050 val / 1,750 test on the same chronological split). Both models now clearly separate from the naive baseline: LightGBM log loss 1.005, Logistic Regression 1.006, baseline 1.067 — accuracy up to ~50–52% from the low 40s. LightGBM narrowly won this run (previously logistic regression had won on the smaller dataset) — `active_model.json` and `app.py`'s dynamic loader handled the flip with no code changes needed, which is exactly what that refactor was for.
- End-to-end re-tested locally (`/health`, `/matchweeks`, `/predictions`, `/team/{id}`) — all serving correctly against the new gold table.

### Open / next (updated)
- [ ] Kaggle CSV wired in for the LOCAL pipeline only — the Databricks (PySpark/Delta) notebooks still use the old 2-season, id-only approach and have NOT been updated to match (would need the same team_key join logic ported to Spark, plus landing the CSV into a Delta bronze table).
- [ ] `/h2h` endpoint still matches on numeric `team_id`, so it only surfaces API-era (2023-24+) meetings between two teams, not the much deeper Kaggle-era history — a reasonable follow-up would be to also match on `team_key` there for teams that have one.
- [x] Fixed before it could bite: the Kaggle CSV was originally landed under `data/bronze/`, which is entirely gitignored - moved it to a new tracked path, `external_data/kaggle_epl_history.csv` (~730KB, committed on purpose), and pointed `silver_transform.py` at it there, so Render/Databricks deploys see the same 25-season depth as local dev instead of silently falling back to just the 4 API seasons.
- [ ] Push `test` branch from PowerShell (device shell still can't do interactive GitHub OAuth) and confirm Render redeploys with the deeper dataset.


### 2026-09-08 (later still): manager-tenure feature, and why player-level data was left out

User asked for player and manager "form" data. Investigated three
scraping targets before touching any code:

- **Transfermarkt** - explicitly disallows scraping in its terms of
  service and runs active anti-bot protection to enforce it. Declined
  on principle (not just difficulty) - going around a site's ToS isn't
  something this project does, "no cut corners" cuts both ways.
- **Understat** - `robots.txt` is `Disallow: /` for every user agent.
  Respected it, moved on.
- **FBref (Sports Reference)** - Cloudflare bot-challenge on every
  request; same category as Transfermarkt.
- **API-Football (api-football.com/api-sports.io)** - a real, legitimate
  API (not scraping) with a free tier covering player stats, injuries,
  lineups and coaches. Flagged as a genuine option for later: needs the
  user's own signup + API key, and the free tier (100 req/day) means a
  multi-day backfill, not a same-day addition. Parked for now.

**Shipped:** manager tenure, scraped from Wikipedia's "List of Premier
League managers" page (CC BY-SA, robots.txt-permitted, no anti-bot
wall) - `src/ingest_managers.py` pulls the full appointment history
table (508 appointments, 51 clubs, back to 1992) into
`external_data/manager_tenures.csv` (tracked in git, same reasoning as
the Kaggle CSV). Chose this over the Kaggle CSV's extra shot/corner/card
columns for the same reason those were rejected earlier: a feature has
to work for LIVE predictions, not just backfilled history, and Wikipedia's
table includes ongoing tenures so it covers both.

- New pre-match features: `home_manager_tenure_days` / `away_manager_tenure_days` (days the manager has been in charge as of kickoff - a real, documented effect in football analytics, the "new manager bounce"). Joined via the same `team_key` as-of logic as rolling form; no leakage risk since only each manager's start date is used.
- Coverage check before trusting it: all 27 current-era team_keys matched the Wikipedia table; 98.1% of matches fell inside a known tenure window (the ~2% gap is mostly short caretaker spells with unparseable Wikipedia dates, which the loader correctly treats as missing rather than guessing).
- Result: modest, real improvement - LightGBM log loss 1.005 → 1.000, Logistic Regression 1.006 → 1.003, accuracy up to 0.523 (logreg) / 0.503 (lgbm) from 0.517/0.503. LightGBM still wins this run.
- Re-verified `/health` and `/predictions` serve correctly with the new feature set.

**Explicitly not done:** individual player-level form/injury data. No
free source exists that's both ToS-compliant and usable for live
(not just historical) predictions - this is being honest about a real
limitation rather than faking placeholder data. API-Football is the
legitimate path if the user wants to invest the multi-day backfill time.


### 2026-09-08 (later still): Databricks pipeline brought back up to date

The Databricks notebooks had fallen behind the three local-pipeline
upgrades from earlier today (season expansion, Kaggle merge, manager
tenure, dual-model comparison). Ported all three over to PySpark:

- `01_bronze_ingest.py`: season list extended to match local (2023-24 through current, i.e. `[2025, 2024, 2023]` trailing).
- `02_silver_transform.py`: now unions football-data.org rows with the Kaggle CSV (season_start_year < 2023 only, same no-overlap rule as local). The Kaggle CSV and manager_tenures.csv were uploaded to the workspace as Workspace Files (`/Workspace/Shared/pl_predictor/external_data/`) via the REST API, since Unity Catalog blocks DBFS but not `/Workspace` paths - read with plain pandas on the driver, same trick as before. Team-name normalization (`team_key`) is a Spark UDF, duplicated from `src/team_names.py` rather than imported (no clean cross-environment import path for a plain module on this serverless workspace).
- `03_gold_features.py`: rolling form and head-to-head rekeyed from `team_id` to `team_key` (window functions, same as local's pandas rewrite). Manager tenure added as a broadcast range-condition join (start_date <= match_date <= end_date) against the small Wikipedia-scraped lookup table - conceptually the Spark equivalent of the local `merge_asof`.
- `04_train_model.py`: now trains both LightGBM and Logistic Regression on the identical split and picks the winner by test log loss, matching local. The base64-to-Delta-table publish step (`pl_predictor_gold.latest_model`, the workaround for this Unity Catalog workspace blocking direct artifact download) now handles either model type.
- **Bug caught during the first job run:** `spark.createDataFrame([model_row])` failed with `CANNOT_DETERMINE_TYPE` - Spark can't infer a schema when a dict column is `None` for every row, which is always true here (whichever model type didn't win has all-null serialization columns). Fixed with an explicit `StructType` schema instead of relying on inference.
- Re-ran end-to-end via `runs/submit` on serverless job compute (same approach as the original validation) - **succeeded**. Queried the published Delta table via the SQL Warehouse: 9,820 finished matches, LightGBM log loss 1.0014 / logistic regression 1.0022, both beating the 1.0675 baseline - matches the local/Render numbers almost exactly.

Databricks and the local/Render pipeline are now in sync on data and methodology (still two independently-trained models, as designed - not a shared artifact - but same features, same seasons, same comparison logic).

### On scraping requests: FBref

User asked about scraping fbref.com for player data. Declined for the
same reason as Transfermarkt: FBref (Sports Reference) runs an active
Cloudflare bot-challenge on requests (confirmed by testing `robots.txt`
directly - it returned a JS challenge page, not the file) and Sports
Reference's terms of service explicitly prohibit scraping/automated
collection without a license. Pointed back to API-Football as the
legitimate path; user is signing up for a free API key.


### 2026-09-08 (later still): player-season stats from user-supplied FBref data

User provided `PLAYERDATA.txt` (923KB) - 14 pasted "Player Standard
Stats" tables from FBref/Sports Reference, 2015-16 through 2026-27,
copy-pasted by the user themselves (not scraped by this session -
important distinction after the earlier Transfermarkt/FBref-scraping
declines: this data was already in hand, obtained by the user directly
through the site's normal page view, and FBref's own stated terms
("please cite us and provide a link and/or a mention") anticipate this
kind of use - a citation was added to README.md's new Data sources
section rather than skipped).

**Cleaning (`src/ingest_player_stats.py`):**
- 14 season blocks, 2 were exact-duplicate pastes (2025-26, 2016-17 each appeared twice, byte-identical) - deduped: 7,500 raw rows -> 6,406 clean rows.
- FBref's Age column format changes between exports - plain years ("25") in older pulls, years-days ("29-335") in newer ones - normalized to a single float.
- Fixed a genuine mislabel before it shipped: the "G+A" (goals+assists combined) column was initially named `ga`, which reads like "goals against" - renamed to `g_plus_a` for clarity; the data itself was never wrong, just the column name.
- Squad names checked against the existing team_key mapping: 2 of 35 didn't match ("Manchester Utd", a truncated "Nottingham") - extended team_names.py's alias table rather than silently dropping those two clubs' data.
- Output: `external_data/player_season_stats.csv`, tracked in git (same reasoning as the other external sources).

**Leakage-safe design:** a team's CURRENT season stats can't be used mid-season (a May goal tally "knows" about matches that haven't happened yet in August) - so gold_features.py aggregates player rows to squad-season totals and joins the PRIOR completed season's totals onto every match in the following season instead. New enrichment features: `{home,away}_prev_season_goals`, `{home,away}_prev_season_avg_age`, `{home,away}_prev_season_squad_size`.

**Coverage forced a real design change.** Checked before assuming it'd just work: squad prior-season data only covers 33% of all finished matches (0% before 2016-17, since that's as far back as FBref's export goes; ~85% from 2016-17 onward). Blanket-requiring it via the existing `dropna(subset=PRE_MATCH_FEATURES)` pattern would have thrown out roughly two-thirds of the hard-won training set. Fixed properly rather than working around it: split every pre-match feature into `CORE_FEATURES` (rolling form, rest days - missing these genuinely means "no history," rows still dropped) and `ENRICHMENT_FEATURES` (h2h, manager tenure, squad prior-season stats - missing these gets median-imputed using ONLY the training split's median, computed after the time-based split so no leakage). Both lists now live in `gold_features.py` and are imported by `train_model.py` and `app.py`, instead of being hand-duplicated in three files (the actual root cause of an earlier near-miss where `app.py`'s feature list could have drifted out of sync).

**Result:** training set grew to 9,748 usable rows (6,628 train / 1,170 val / 1,950 test) - up from 8,747 even before this change, since manager-tenure rows that used to get dropped under the old blanket-dropna are now recovered too. LightGBM log loss 1.002, logistic regression 1.013, both beat the 1.067 baseline; LightGBM stayed the active model. Re-verified `/health`, `/matchweeks`, `/predictions` all serve correctly.

**Not yet done:** the Databricks notebooks were not updated for this change (still on the manager-tenure-only feature set) - same kind of drift as before, logged as an open item rather than silently left unmentioned.


### 2026-09-08 (later still): player impact tiers - classification, model features, API, frontend

User's request: "the teams and players form is very important for the
last 10 years... lets use 10 years of player form and track each player
and classify players into 5 categories based on their impact in their
own team so users can see that and it can be used in prediction." Two
deliverables implied - a feature for the model, and something visible
to users - both done.

**Classification (`src/ingest_player_stats.py`, `classify_impact()`):**
`impact_score = minutes_share * (1 + 0.5 * non_penalty_G+A_per90)`, where
`minutes_share` = a player's 90s played / their squad's total 90s that
season - the dominant term on purpose, since minutes trusted by the
manager is the best available proxy for "how central is this player to
this team," and per-90 output is a secondary multiplier rather than a
substitute (a nailed-on, low-scoring centre-back should still rank
above a mop-up-minutes player with a couple of garbage-time goals).
Ranked within `(squad_key, season_start_year)` only - never across
squads, so a "Tier 1" at a relegation candidate isn't claimed to be
equal to a "Tier 1" at a title contender, just equally central to their
own team that season. Split into 5 equal-count tiers by rank (not
`qcut` - broke on the many tied/near-zero scores from single-appearance
players): 1 Talisman, 2 Key Player, 3 Regular Starter, 4 Squad
Rotation, 5 Fringe/Backup. 6,406 rows -> tier distribution roughly even
(1182/1285/1276/1285/1378). Spot-checked against known players: Salah
and Haaland come back Tier 1 in their peak seasons; van Dijk correctly
drops to Tier 4 (Squad Rotation) specifically during his 2020-21
injury-recovery season, not any other - the minutes-share design is
doing what it's supposed to.

**New model features (`src/gold_features.py`, `load_squad_prev_season_stats()`):**
added `{home,away}_prev_season_n_stars` (count of Tier 1/2 players on
last season's squad) and `{home,away}_prev_season_star_power` (sum of
impact_score across the squad) to `ENRICHMENT_FEATURES`, joined the
same leakage-safe way as the other prior-season squad aggregates
(prior COMPLETED season only, median-imputed pre-2016-17 where FBref
data doesn't exist). Re-ran the full pipeline: gold layer 10,170 rows
(9,820 finished), enrichment coverage h2h 89.1%, manager_tenure 98.1%,
all prev_season_* features 33.1-33.4% (unchanged - same coverage
ceiling as before, just two more columns riding along). Training:
LightGBM log loss 1.003 / acc 0.507, logistic regression log loss
1.019 / acc 0.502, both still beat the 1.067 baseline; split
6,628/1,170/1,950 train/val/test.

**New API endpoint (`src/app.py`, `GET /team/{team_id}/impact`):**
looks up a team's `team_key`, finds its most recent season in
`player_season_stats.csv`, returns the full squad sorted by
`team_rank` with tier/tier_label plus basic stats (goals, assists,
90s played). Tested end-to-end against a live uvicorn instance -
`/team/64/impact` (Liverpool) returned Isak/Wirtz/Gakpo as Talismans
and van Dijk as Key Player for the current 2026-27 season, which
matches the small early-season sample sensibly.

**Frontend (`static/index.html`):** added a third "Player Impact" tab
next to "Head-to-head"/"Squads" in the fixture modal - two columns
(home/away), each player shown with a tier badge (color-coded by tier)
plus goals/assists/90s-played, loaded from the new endpoint the same
way the existing Squads tab loads `/team/{id}`. This is the "so users
can see that" half of the request - the model-feature half was already
covered by n_stars/star_power above.

**Not yet done:** Databricks notebooks still don't have the
player-impact features (same drift-tracking note as the prior entry -
now two features behind: manager-tenure-only feature set on Databricks
vs. h2h + manager tenure + prior-season squad + star-power locally).
Logged, not silently skipped - not touched this round without the user
asking for it specifically.
