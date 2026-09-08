# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2a - Silver layer (PySpark + Delta Lake)
# MAGIC
# MAGIC Reads the bronze `matches` Delta table (raw JSON strings, football-data.org,
# MAGIC 2023-24 season onward), parses/flattens it, AND unions in the Kaggle
# MAGIC historical CSV (2000/01-2022/23 - seasons the free API tier can't reach)
# MAGIC so the gold layer has real depth to train on instead of ~3 seasons.
# MAGIC
# MAGIC The two sources don't share a team id space (football-data.org has
# MAGIC numeric ids, Kaggle only has club names), so every row also gets a
# MAGIC normalized `team_key` string - this is what 03_gold_features.py joins
# MAGIC rolling form/head-to-head on, so a club's Kaggle-era and API-era
# MAGIC matches feed the SAME calculation. Mirrors src/team_names.py and
# MAGIC src/silver_transform.py in the local pipeline exactly.

# COMMAND ----------

import re

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType
)

BRONZE_DB = "pl_predictor_bronze"
SILVER_DB = "pl_predictor_silver"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {SILVER_DB}")

# The API's free tier only reaches this season and later - Kaggle fills
# in everything strictly before it, so there's no double-counted overlap.
API_FLOOR_SEASON = 2023
KAGGLE_CSV_PATH = "/Workspace/Shared/pl_predictor/external_data/kaggle_epl_history.csv"

match_schema = StructType([
    StructField("id", LongType()),
    StructField("utcDate", StringType()),
    StructField("status", StringType()),
    StructField("matchday", IntegerType()),
    StructField("homeTeam", StructType([
        StructField("id", LongType()), StructField("name", StringType())
    ])),
    StructField("awayTeam", StructType([
        StructField("id", LongType()), StructField("name", StringType())
    ])),
    StructField("score", StructType([
        StructField("winner", StringType()),
        StructField("fullTime", StructType([
            StructField("home", IntegerType()), StructField("away", IntegerType())
        ])),
    ])),
])

# COMMAND ----------

# --- team name normalization (mirrors src/team_names.py - duplicated here
# rather than imported, since %run only chains other notebooks and this
# project's serverless workspace doesn't have a clean path to import a
# plain .py module across both the local repo and Databricks) ---

_ALIASES = {
    "MANCITY": "MANCHESTERCITY", "MANUNITED": "MANCHESTERUNITED", "MANUTD": "MANCHESTERUNITED",
    "SPURS": "TOTTENHAMHOTSPUR", "TOTTENHAM": "TOTTENHAMHOTSPUR", "WOLVES": "WOLVERHAMPTONWANDERERS",
    "NOTTMFOREST": "NOTTINGHAMFOREST", "NOTTINGHAMFOREST": "NOTTINGHAMFOREST",
    "WESTBROM": "WESTBROMWICHALBION", "WESTBROMWICH": "WESTBROMWICHALBION",
    "WESTHAM": "WESTHAMUNITED", "NEWCASTLE": "NEWCASTLEUNITED", "LEICESTER": "LEICESTERCITY",
    "LEEDS": "LEEDSUNITED", "NORWICH": "NORWICHCITY", "STOKE": "STOKECITY",
    "SWANSEA": "SWANSEACITY", "HULL": "HULLCITY", "CARDIFF": "CARDIFFCITY",
    "BIRMINGHAM": "BIRMINGHAMCITY", "CHARLTON": "CHARLTONATHLETIC", "BOLTON": "BOLTONWANDERERS",
    "BLACKBURN": "BLACKBURNROVERS", "BRADFORD": "BRADFORDCITY", "COVENTRY": "COVENTRYCITY",
    "DERBY": "DERBYCOUNTY", "IPSWICH": "IPSWICHTOWN", "HUDDERSFIELD": "HUDDERSFIELDTOWN",
    "BRIGHTON": "BRIGHTONHOVEALBION", "QPR": "QUEENSPARKRANGERS", "WIGAN": "WIGANATHLETIC",
    "BOURNEMOUTH": "AFCBOURNEMOUTH", "LUTON": "LUTONTOWN",
}
_SUFFIX_RE = re.compile(r"\b(FC|AFC)\b")
_NONALNUM_RE = re.compile(r"[^A-Z0-9]")


def normalize_team_name(name):
    if not name:
        return ""
    key = name.upper()
    key = _SUFFIX_RE.sub("", key)
    key = _NONALNUM_RE.sub("", key)
    return _ALIASES.get(key, key)


normalize_udf = F.udf(normalize_team_name, StringType())

# COMMAND ----------

# --- API-sourced rows (2023-24+) ---

bronze = spark.table(f"{BRONZE_DB}.matches")

latest = (
    bronze.withColumn(
        "rn",
        F.row_number().over(
            __import__("pyspark.sql.window", fromlist=["Window"]).Window
            .partitionBy("match_id")
            .orderBy(F.col("ingested_at").desc())
        ),
    )
    .filter("rn = 1")
    .drop("rn")
)

parsed = latest.withColumn("m", F.from_json("raw_json", match_schema))

api_silver = parsed.select(
    F.col("m.id").alias("match_id"),
    F.to_timestamp("m.utcDate").alias("utc_date"),
    F.col("m.status").alias("status"),
    F.col("m.matchday").alias("matchday"),
    F.col("m.homeTeam.id").alias("home_team_id"),
    F.col("m.homeTeam.name").alias("home_team_name"),
    normalize_udf(F.col("m.homeTeam.name")).alias("home_team_key"),
    F.col("m.awayTeam.id").alias("away_team_id"),
    F.col("m.awayTeam.name").alias("away_team_name"),
    normalize_udf(F.col("m.awayTeam.name")).alias("away_team_key"),
    F.col("m.score.fullTime.home").alias("home_goals"),
    F.col("m.score.fullTime.away").alias("away_goals"),
    F.col("m.score.winner").alias("winner"),
    F.col("season_start_year").alias("season_start_year"),
    F.lit("api").alias("source"),
)

# COMMAND ----------

# --- Kaggle historical rows (2000/01-2022/23 only, no overlap with the API) ---
# Small static CSV (~9.4k rows) - processed with pandas on the driver
# (a Workspace File, read directly with plain Python I/O - dbutils.fs/DBFS
# access is blocked on this Unity Catalog serverless workspace, but
# /Workspace paths are not), then handed to Spark as a DataFrame so it
# joins the rest of the pipeline the same way the API rows do.

import hashlib

kg_pd = pd.read_csv(KAGGLE_CSV_PATH)
kg_pd["season_start_year"] = kg_pd["Season"].str.slice(0, 4).astype(int)
kg_pd = kg_pd[kg_pd["season_start_year"] < API_FLOOR_SEASON].copy()

_RESULT_TO_WINNER = {"H": "HOME_TEAM", "A": "AWAY_TEAM", "D": "DRAW"}


def synth_match_id(row):
    raw = f"{row['MatchDate']}|{row['HomeTeam']}|{row['AwayTeam']}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:8]
    return -int(digest, 16)


kaggle_rows = pd.DataFrame({
    "match_id": kg_pd.apply(synth_match_id, axis=1),
    "utc_date": pd.to_datetime(kg_pd["MatchDate"], utc=True),
    "status": "FINISHED",
    "matchday": None,
    "home_team_id": None,
    "home_team_name": kg_pd["HomeTeam"],
    "home_team_key": kg_pd["HomeTeam"].apply(normalize_team_name),
    "away_team_id": None,
    "away_team_name": kg_pd["AwayTeam"],
    "away_team_key": kg_pd["AwayTeam"].apply(normalize_team_name),
    "home_goals": kg_pd["FullTimeHomeGoals"].astype(int),
    "away_goals": kg_pd["FullTimeAwayGoals"].astype(int),
    "winner": kg_pd["FullTimeResult"].map(_RESULT_TO_WINNER),
    "season_start_year": kg_pd["season_start_year"].astype(int),
    "source": "kaggle",
})
# object dtype None columns confuse Spark's schema inference - cast explicitly
kaggle_rows["matchday"] = kaggle_rows["matchday"].astype("Int64")
kaggle_rows["home_team_id"] = kaggle_rows["home_team_id"].astype("Int64")
kaggle_rows["away_team_id"] = kaggle_rows["away_team_id"].astype("Int64")

kaggle_silver = spark.createDataFrame(kaggle_rows).select(
    F.col("match_id").cast(LongType()),
    F.col("utc_date").cast("timestamp"),
    "status", "matchday",
    F.col("home_team_id").cast(LongType()), "home_team_name", "home_team_key",
    F.col("away_team_id").cast(LongType()), "away_team_name", "away_team_key",
    F.col("home_goals").cast(IntegerType()), F.col("away_goals").cast(IntegerType()),
    "winner", "season_start_year", "source",
)

# COMMAND ----------

silver = api_silver.unionByName(kaggle_silver).orderBy("utc_date")

silver.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{SILVER_DB}.matches"
)

n_finished = silver.filter("status = 'FINISHED'").count()
n_api = silver.filter("source = 'api'").count()
n_kaggle = silver.filter("source = 'kaggle'").count()
print(f"Silver matches table: {silver.count()} rows ({n_finished} finished) -> {SILVER_DB}.matches")
print(f"  -> {n_api} from football-data.org (2023-24+), {n_kaggle} from Kaggle history (pre-2023-24)")
