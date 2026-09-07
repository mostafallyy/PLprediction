# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2a - Silver layer (PySpark + Delta Lake)
# MAGIC
# MAGIC Reads the bronze `matches` Delta table (raw JSON strings), parses
# MAGIC and flattens into one clean, typed row per match, and writes a
# MAGIC silver Delta table. Pure clean/join here - no feature engineering.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType
)

BRONZE_DB = "pl_predictor_bronze"
SILVER_DB = "pl_predictor_silver"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {SILVER_DB}")

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

bronze = spark.table(f"{BRONZE_DB}.matches")

# take the most recent pull only (dedupe by match_id, keep latest ingested_at)
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

silver = parsed.select(
    F.col("m.id").alias("match_id"),
    F.to_timestamp("m.utcDate").alias("utc_date"),
    F.col("m.status").alias("status"),
    F.col("m.matchday").alias("matchday"),
    F.col("m.homeTeam.id").alias("home_team_id"),
    F.col("m.homeTeam.name").alias("home_team_name"),
    F.col("m.awayTeam.id").alias("away_team_id"),
    F.col("m.awayTeam.name").alias("away_team_name"),
    F.col("m.score.fullTime.home").alias("home_goals"),
    F.col("m.score.fullTime.away").alias("away_goals"),
    F.col("m.score.winner").alias("winner"),
).orderBy("utc_date")

silver.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{SILVER_DB}.matches"
)

n_finished = silver.filter("status = 'FINISHED'").count()
print(f"Silver matches table: {silver.count()} rows ({n_finished} finished) -> {SILVER_DB}.matches")
