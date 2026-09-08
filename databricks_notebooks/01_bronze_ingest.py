# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 1 - Bronze ingestion (PySpark + Delta Lake)
# MAGIC
# MAGIC Pulls fixtures/results/standings from football-data.org and lands the
# MAGIC raw JSON as a Delta table in the `bronze` schema - untouched, one row
# MAGIC per API pull, so we can always replay history if silver/gold logic
# MAGIC changes later.
# MAGIC
# MAGIC Set `FOOTBALL_DATA_API_KEY` as a Databricks secret
# MAGIC (`databricks secrets put-secret pl_predictor football_data_api_key`)
# MAGIC or widget parameter before running.

# COMMAND ----------

import json
from datetime import datetime, timezone

import requests
from pyspark.sql import Row
from pyspark.sql.functions import current_timestamp, lit

dbutils.widgets.text("api_key", "", "football-data.org API key")
API_KEY = dbutils.widgets.get("api_key") or dbutils.secrets.get("pl_predictor", "football_data_api_key")

CATALOG_DB = "pl_predictor_bronze"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {CATALOG_DB}")

API_BASE = "https://api.football-data.org/v4"
COMPETITION = "PL"
CURRENT_SEASON_START_YEAR = 2026
TRAIN_SEASON_START_YEARS = [2025, 2024, 2023]  # last 3 completed seasons available on the free tier (2022 and earlier return 403) - matches src/ingest_bronze.py

# COMMAND ----------

def fetch(endpoint, params=None):
    resp = requests.get(
        f"{API_BASE}/{endpoint}",
        headers={"X-Auth-Token": API_KEY},
        params=params or {},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


run_ts = datetime.now(timezone.utc)
run_date = run_ts.strftime("%Y-%m-%d")

# --- matches: current season + last 3 completed seasons, merged ---
all_matches = []
seen_ids = set()
for season in [CURRENT_SEASON_START_YEAR] + TRAIN_SEASON_START_YEARS:
    payload = fetch(f"competitions/{COMPETITION}/matches", params={"season": season})
    for m in payload.get("matches", []):
        if m["id"] not in seen_ids:
            m["_season_start_year"] = season
            m["_raw_json"] = json.dumps(m)
            all_matches.append(m)
            seen_ids.add(m["id"])
    print(f"season {season}: {len(payload.get('matches', []))} matches")

matches_rows = [
    Row(match_id=m["id"], season_start_year=m["_season_start_year"], raw_json=m["_raw_json"])
    for m in all_matches
]
matches_df = (
    spark.createDataFrame(matches_rows)
    .withColumn("pull_date", lit(run_date))
    .withColumn("ingested_at", current_timestamp())
)

(
    matches_df.write.format("delta")
    .mode("append")
    .partitionBy("pull_date")
    .saveAsTable(f"{CATALOG_DB}.matches")
)
print(f"Landed {matches_df.count()} match records -> {CATALOG_DB}.matches")

# --- standings + teams: current season only, for context ---
for name, endpoint in {
    "standings": f"competitions/{COMPETITION}/standings",
    "teams": f"competitions/{COMPETITION}/teams",
}.items():
    payload = fetch(endpoint)
    row_df = (
        spark.createDataFrame([Row(raw_json=json.dumps(payload))])
        .withColumn("pull_date", lit(run_date))
        .withColumn("ingested_at", current_timestamp())
    )
    (
        row_df.write.format("delta")
        .mode("append")
        .partitionBy("pull_date")
        .saveAsTable(f"{CATALOG_DB}.{name}")
    )
    print(f"Landed {name} -> {CATALOG_DB}.{name}")

print("Bronze ingestion complete.")
