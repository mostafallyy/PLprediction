# Premier League Match Predictor

A live-data pipeline that predicts Premier League match outcomes: daily
ingestion, a PySpark medallion pipeline (bronze/silver/gold) on
Databricks with Delta Lake, a leakage-checked LightGBM model tracked in
MLflow, and a FastAPI service that returns win/draw/loss probabilities
for upcoming fixtures.

## Data sources

- [football-data.org](https://www.football-data.org/) - live fixtures, results, standings, team crests (free tier, 2023-24 season onward).
- [Kaggle: EPL Match Data 2000-2025](https://www.kaggle.com/datasets/marcohuiii/english-premier-league-epl-match-data-2000-2025) (marcohuiii) - historical match results, 2000/01 through 2022-23, filling the gap the free API tier can't reach.
- Wikipedia, [List of Premier League managers](https://en.wikipedia.org/wiki/List_of_Premier_League_managers) (CC BY-SA) - full managerial appointment history, used for the manager-tenure feature.
- FBref / Sports Reference - player-season Standard Stats, Miscellaneous Stats (fouls, tackles won, interceptions), and Goalkeeping tables, 2015-16 through 2026-27, used for the prior-season squad-strength feature AND for classifying every player-season into a 1-5 impact tier (Talisman -> Fringe/Backup) shown in the app's "Player Impact" tab. Per FBref's data usage terms, this project credits Sports Reference (Data via [fbref.com](https://fbref.com), Sports Reference LLC) - only season-level aggregates are used, and only the immediately PRIOR completed season's totals are ever joined onto a match, never the match's own season (see `src/ingest_player_stats.py` for the leakage reasoning and the tier-classification methodology, including position-aware attack/defense weighting and small-sample shrinkage).

## Architecture

```
football-data.org API
        |
        v
  Bronze  (Databricks, Delta) <- raw JSON, untouched, partitioned by pull date
        |
        v
  Silver  (Databricks, Delta) <- cleaned, typed, one row per match
        |
        v
  Gold    (Databricks, Delta) <- pre-match features, tagged safe/leakage,
                                  built for BOTH finished and upcoming matches
        |
        v
  Model   (Databricks, MLflow) <- LightGBM, time-based train/val/test split,
                                   early stopping
        |
        v
  Serving (local FastAPI, src/app.py) <- pulls the trained model + gold
                                          features down and serves /predictions
```

Everything from raw API pull through model training runs as real PySpark
on Databricks (`databricks_notebooks/`), wired into a single scheduled
Databricks Job (`pl_predictor_daily_pipeline`, daily at 06:00 UTC,
currently paused - enable it in the Databricks Jobs UI to go live). A
lightweight local pipeline (`src/`, plain pandas) exists as a dev/offline
mode for iterating on logic without touching the workspace, and is what
`run_pipeline.sh` runs directly.

## Databricks notebooks (`databricks_notebooks/`)

| Notebook | Does |
|---|---|
| `01_bronze_ingest.py` | Pulls matches/standings/teams from football-data.org (API key from a Databricks secret), lands raw JSON as a Delta table |
| `02_silver_transform.py` | Parses/cleans into one typed row per match |
| `03_gold_features.py` | Builds pre-match features with Spark window functions - see below |
| `04_train_model.py` | LightGBM, time-based split, MLflow-tracked, publishes the model to `pl_predictor_gold.latest_model` |
| `00_run_all.py` | Chains all four via `%run`, for a manual one-click run |

### The as-of join, without `merge_asof`

Spark has no `pandas.merge_asof`. The gold notebook rebuilds "this
team's form strictly before this match" (needed for BOTH finished
results and future fixtures) with window functions: compute each team's
rolling form as of right after every finished match, attach that
snapshot back onto its own row, then carry it forward with
`last(..., ignoreNulls=True)` over `rowsBetween(unboundedPreceding, -1)`
- "the last known snapshot from a strictly earlier row" - onto every
row, finished or scheduled.

### Leakage discipline

Every gold-layer column is tagged `pre_match` (safe to train/serve on)
or `post_match` (kept only for the label - final score, winner - never
fed to the model). The train/val/test split is time-based, not random.

## Data

Early in a new PL season there aren't enough finished matches to train
on, so ingestion pulls the current season (for fixtures to predict) plus
the last two completed seasons (for training history).

## Validation numbers (Databricks-trained, honest, not cherry-picked)

| Metric | Model | Naive baseline |
|---|---|---|
| Accuracy | ~0.48 | 0.43 (always predict majority class) |
| Log loss | ~1.07 | 1.08 (predict training class frequencies) |

A modest but real lift over guessing - expected for ~300 training
matches, 11 features, no player-level data. Re-trained daily; re-measure
periodically rather than trusting this as a permanent number.

## Running the Databricks pipeline

Requires a Databricks workspace (Community Edition/Free Edition works -
this was built and validated against a serverless free-tier workspace)
and a personal access token.

```bash
export DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com
export DATABRICKS_TOKEN=<your-token>

# one-time setup
databricks secrets create-scope pl_predictor
databricks secrets put-secret pl_predictor football_data_api_key --string-value <your-football-data-key>
databricks workspace import-dir databricks_notebooks /Workspace/Shared/pl_predictor

# ad-hoc run of one notebook, e.g. via the Jobs API runs/submit,
# or open 00_run_all in the workspace UI and Run All
```

The daily schedule is a Databricks Job (`pl_predictor_daily_pipeline`,
06:00 UTC) chaining all four notebooks with task dependencies -
enable/disable it from the Jobs UI.

### Note on Community/Free Edition compute

This workspace has no classic clusters ("no associated worker
environments") - it's serverless-only. Jobs run fine on serverless job
compute; `dbutils.fs.cp` to `dbfs:/` and the raw MLflow-artifact/DBFS
REST proxy are blocked under Unity Catalog, so the trained model is
published to a Delta table (`pl_predictor_gold.latest_model`, model
bytes as base64) instead - queryable through the SQL Warehouse, which
is how the local FastAPI service pulls it down.

## Running the local (pandas) dev pipeline

```bash
pip install -r requirements.txt
export FOOTBALL_DATA_API_KEY=your_key   # https://www.football-data.org/client/register

bash run_pipeline.sh                     # ingest -> silver -> gold -> train
uvicorn src.app:app --reload --port 8000
curl http://localhost:8000/predictions
```

Or with Docker:

```bash
docker build -t pl-predictor .
docker run -p 8000:8000 pl-predictor
```

## What's next

- Deploy the FastAPI container to Render for a public URL
- Add a dashboard (Streamlit or Power BI) over `/predictions`
- Player-level features (injuries, suspensions)
- Multi-league expansion
- Compare against a simple Elo-rating baseline to quantify model lift
