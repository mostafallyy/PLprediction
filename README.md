# Premier League Match Predictor

A live-data pipeline that predicts Premier League match outcomes: daily
ingestion, a medallion-architecture transform layer, a leakage-checked
LightGBM model tracked in MLflow, and a FastAPI service that returns
win/draw/loss probabilities for upcoming fixtures.

## Problem

Predict the outcome (home win / draw / away win) of Premier League
matches before kickoff, using only information that would genuinely be
available in advance - recent form, home/away splits, head-to-head
history, and rest days. No post-match data (final score, etc.) is ever
allowed to reach the model; see "Leakage discipline" below.

## Architecture

```
football-data.org API
        |
        v
  Bronze  (data/bronze)   <- raw JSON, untouched, partitioned by pull date
        |
        v
  Silver  (data/silver)   <- cleaned, typed, one row per match
        |
        v
  Gold    (data/gold)     <- pre-match features, tagged safe/leakage,
                              built for BOTH finished and upcoming matches
        |
        v
  Model   (models/)       <- LightGBM, time-based train/val/test split,
                              early stopping, MLflow-tracked
        |
        v
  Serving (src/app.py)    <- FastAPI /predictions endpoint, Dockerized
```

`run_pipeline.sh` chains ingest -> silver -> gold -> train into one
command; `.github/workflows/daily_pipeline.yml` runs it daily and
commits the refreshed gold table and model.

## Leakage discipline

Every gold-layer column is tagged in `src/gold_features.py` as either
`pre_match` (safe to train/serve on) or `post_match` (kept only for
building the label, e.g. final score - never fed to the model). Rolling
form is computed with `pandas.merge_asof(..., allow_exact_matches=False)`
so a match can only see form built from matches strictly BEFORE it,
including for upcoming fixtures.

The train/val/test split is **time-based, not random**: the model
trains on the earliest matches, validates (for early stopping) on the
next slice, and is scored on the most recent, held-out slice - never a
random shuffle, which would leak future form into the past.

## Data

Because this is early in the current PL season, the current season
alone doesn't yet have enough finished matches to train on. The
ingestion script pulls the current season (for upcoming fixtures to
predict) plus the last two completed seasons (for training history),
merged into one bronze table.

## Validation numbers (honest, not cherry-picked)

Measured on a held-out, time-ordered test slice (most recent matches),
against two naive baselines:

| Metric | Model | Naive baseline |
|---|---|---|
| Accuracy | ~0.49 | 0.44 (always predict majority class) |
| Log loss | ~1.07 | 1.08 (predict training class frequencies) |

This is a modest but real lift over guessing - expected for an early
baseline with ~300 training matches and 11 features, no player-level
data, and no tuning beyond basic regularization + early stopping. The
model is re-trained daily as more matches finish, and this table should
be re-measured periodically rather than trusted as a permanent number.

## Running it

```bash
pip install -r requirements.txt
export FOOTBALL_DATA_API_KEY=your_key   # https://www.football-data.org/client/register

bash run_pipeline.sh                     # ingest -> silver -> gold -> train
uvicorn src.app:app --reload --port 8000 # serve predictions
curl http://localhost:8000/predictions
```

Or with Docker:

```bash
docker build -t pl-predictor .
docker run -p 8000:8000 pl-predictor
```

## What's next

- Deploy the container to Render for a public URL
- Add a dashboard (Streamlit or Power BI) over `/predictions`
- Player-level features (injuries, suspensions)
- Multi-league expansion
- Compare against a simple Elo-rating baseline to quantify model lift
