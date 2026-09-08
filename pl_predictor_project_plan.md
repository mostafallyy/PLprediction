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

