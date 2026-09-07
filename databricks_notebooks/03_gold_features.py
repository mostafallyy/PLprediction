# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2b - Gold feature layer (PySpark + Delta Lake)
# MAGIC
# MAGIC Builds one feature row per match - including SCHEDULED (upcoming)
# MAGIC fixtures - using ONLY information available before kickoff.
# MAGIC
# MAGIC Spark has no `merge_asof`, so the "state of a team strictly before
# MAGIC this match" pattern is rebuilt with window functions:
# MAGIC   1. compute each team's rolling form as of AFTER every FINISHED match
# MAGIC   2. attach that snapshot back onto ITS OWN row (join on team+date)
# MAGIC   3. carry it forward with `last(..., ignoreNulls=True)` over
# MAGIC      `rowsBetween(unboundedPreceding, -1)` - i.e. "the last known
# MAGIC      snapshot from a STRICTLY earlier row" - onto every row,
# MAGIC      finished or scheduled. This is the as-of join, vectorized.
# MAGIC
# MAGIC Every column is tagged pre_match (safe) or post_match (leakage,
# MAGIC label-only) in FEATURE_TAGS, same discipline as the local version.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

SILVER_DB = "pl_predictor_silver"
GOLD_DB = "pl_predictor_gold"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {GOLD_DB}")

FORM_WINDOW = 5

FEATURE_TAGS = {
    "home_form_ppg": "pre_match", "home_form_gf": "pre_match", "home_form_ga": "pre_match",
    "away_form_ppg": "pre_match", "away_form_gf": "pre_match", "away_form_ga": "pre_match",
    "home_home_ppg": "pre_match", "away_away_ppg": "pre_match",
    "h2h_home_win_rate": "pre_match",
    "home_rest_days": "pre_match", "away_rest_days": "pre_match",
    "home_goals": "post_match", "away_goals": "post_match", "winner": "post_match",
}

# COMMAND ----------

matches = spark.table(f"{SILVER_DB}.matches")

home = (
    matches.select(
        "match_id", "utc_date", "status",
        F.col("home_team_id").alias("team_id"),
        F.col("away_team_id").alias("opp_id"),
        F.col("home_goals").alias("gf"),
        F.col("away_goals").alias("ga"),
    ).withColumn("is_home", F.lit(True))
)
away = (
    matches.select(
        "match_id", "utc_date", "status",
        F.col("away_team_id").alias("team_id"),
        F.col("home_team_id").alias("opp_id"),
        F.col("away_goals").alias("gf"),
        F.col("home_goals").alias("ga"),
    ).withColumn("is_home", F.lit(False))
)
team_log = home.unionByName(away)

# COMMAND ----------

# --- Step 1: rolling form as of AFTER each FINISHED match ---
finished = team_log.filter("status = 'FINISHED'").withColumn(
    "pts",
    F.when(F.col("gf") > F.col("ga"), 3)
     .when(F.col("gf") == F.col("ga"), 1)
     .otherwise(0),
)

w_team = Window.partitionBy("team_id").orderBy("utc_date")
w_form = w_team.rowsBetween(-(FORM_WINDOW - 1), 0)
w_expanding = w_team.rowsBetween(Window.unboundedPreceding, 0)

finished_snap = (
    finished
    .withColumn("form_ppg", F.avg("pts").over(w_form))
    .withColumn("form_gf", F.avg("gf").over(w_form))
    .withColumn("form_ga", F.avg("ga").over(w_form))
    .withColumn("home_pts_only", F.when(F.col("is_home"), F.col("pts")))
    .withColumn("away_pts_only", F.when(~F.col("is_home"), F.col("pts")))
    .withColumn("home_ppg_split", F.avg("home_pts_only").over(w_expanding))
    .withColumn("away_ppg_split", F.avg("away_pts_only").over(w_expanding))
    .withColumn("prev_match_date", F.col("utc_date"))
    .select(
        "match_id", "team_id", "utc_date",
        "form_ppg", "form_gf", "form_ga",
        "home_ppg_split", "away_ppg_split", "prev_match_date",
    )
)

# --- Step 2: attach the post-match snapshot back onto its own row ---
combined = team_log.join(finished_snap, on=["match_id", "team_id", "utc_date"], how="left")

# --- Step 3: carry the last STRICTLY PRIOR snapshot forward onto every row ---
w_asof = Window.partitionBy("team_id").orderBy("utc_date").rowsBetween(Window.unboundedPreceding, -1)

team_features = (
    combined
    .withColumn("home_form_ppg_asof", F.last("form_ppg", ignorenulls=True).over(w_asof))
    .withColumn("home_form_gf_asof", F.last("form_gf", ignorenulls=True).over(w_asof))
    .withColumn("home_form_ga_asof", F.last("form_ga", ignorenulls=True).over(w_asof))
    .withColumn("home_ppg_split_asof", F.last("home_ppg_split", ignorenulls=True).over(w_asof))
    .withColumn("away_ppg_split_asof", F.last("away_ppg_split", ignorenulls=True).over(w_asof))
    .withColumn("prev_match_date_asof", F.last("prev_match_date", ignorenulls=True).over(w_asof))
    .withColumn("rest_days", F.datediff("utc_date", "prev_match_date_asof"))
    .select(
        "match_id", "team_id",
        F.col("home_form_ppg_asof").alias("form_ppg"),
        F.col("home_form_gf_asof").alias("form_gf"),
        F.col("home_form_ga_asof").alias("form_ga"),
        F.col("home_ppg_split_asof").alias("home_ppg_split"),
        F.col("away_ppg_split_asof").alias("away_ppg_split"),
        "rest_days",
    )
)

# COMMAND ----------

# --- Head-to-head home-win rate (pair-fixed team_A/team_B trick - see README) ---
pair = (
    matches
    .withColumn("team_a", F.least("home_team_id", "away_team_id"))
    .withColumn("team_b", F.greatest("home_team_id", "away_team_id"))
    .withColumn("is_decisive", (F.col("status") == "FINISHED") & F.col("winner").isin("HOME_TEAM", "AWAY_TEAM"))
    .withColumn(
        "winning_team_id",
        F.when(F.col("winner") == "HOME_TEAM", F.col("home_team_id"))
         .when(F.col("winner") == "AWAY_TEAM", F.col("away_team_id")),
    )
    .withColumn("decisive_flag", F.col("is_decisive").cast("int"))
    .withColumn(
        "team_a_win_flag",
        F.when(F.col("is_decisive") & (F.col("winning_team_id") == F.col("team_a")), 1).otherwise(0),
    )
)

w_pair = Window.partitionBy("team_a", "team_b").orderBy("utc_date").rowsBetween(Window.unboundedPreceding, -1)

h2h = (
    pair
    .withColumn("cum_decisive", F.sum("decisive_flag").over(w_pair))
    .withColumn("cum_team_a_wins", F.sum("team_a_win_flag").over(w_pair))
    .withColumn(
        "h2h_home_win_rate",
        F.when(F.col("cum_decisive") > 0,
               F.when(F.col("home_team_id") == F.col("team_a"), F.col("cum_team_a_wins") / F.col("cum_decisive"))
                .otherwise((F.col("cum_decisive") - F.col("cum_team_a_wins")) / F.col("cum_decisive"))),
    )
    .select("match_id", "h2h_home_win_rate")
)

# COMMAND ----------

home_feat = team_features.select(
    "match_id",
    F.col("form_ppg").alias("home_form_ppg"), F.col("form_gf").alias("home_form_gf"), F.col("form_ga").alias("home_form_ga"),
    F.col("home_ppg_split").alias("home_home_ppg"),
    F.col("rest_days").alias("home_rest_days"),
)
away_feat = team_features.select(
    "match_id",
    F.col("form_ppg").alias("away_form_ppg"), F.col("form_gf").alias("away_form_gf"), F.col("form_ga").alias("away_form_ga"),
    F.col("away_ppg_split").alias("away_away_ppg"),
    F.col("rest_days").alias("away_rest_days"),
)

# team_features has one row per (match_id, team_id) - join home rows on home_team_id, away rows on away_team_id
gold = (
    matches
    .join(
        team_features.select("match_id", F.col("team_id").alias("home_team_id"),
                              F.col("form_ppg").alias("home_form_ppg"), F.col("form_gf").alias("home_form_gf"),
                              F.col("form_ga").alias("home_form_ga"), F.col("home_ppg_split").alias("home_home_ppg"),
                              F.col("rest_days").alias("home_rest_days")),
        on=["match_id", "home_team_id"], how="left",
    )
    .join(
        team_features.select("match_id", F.col("team_id").alias("away_team_id"),
                              F.col("form_ppg").alias("away_form_ppg"), F.col("form_gf").alias("away_form_gf"),
                              F.col("form_ga").alias("away_form_ga"), F.col("away_ppg_split").alias("away_away_ppg"),
                              F.col("rest_days").alias("away_rest_days")),
        on=["match_id", "away_team_id"], how="left",
    )
    .join(h2h, on="match_id", how="left")
    .select(
        "match_id", "utc_date", "home_team_name", "away_team_name", "status",
        "home_form_ppg", "home_form_gf", "home_form_ga",
        "away_form_ppg", "away_form_gf", "away_form_ga",
        "home_home_ppg", "away_away_ppg",
        "h2h_home_win_rate", "home_rest_days", "away_rest_days",
        "home_goals", "away_goals", "winner",
    )
)

gold.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{GOLD_DB}.match_features")

n_finished = gold.filter("status = 'FINISHED'").count()
n_upcoming = gold.filter("status in ('SCHEDULED', 'TIMED')").count()
print(f"Gold feature table: {gold.count()} rows ({n_finished} finished, {n_upcoming} upcoming) -> {GOLD_DB}.match_features")
