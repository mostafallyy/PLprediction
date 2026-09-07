# Databricks notebook source
# MAGIC %md
# MAGIC # Run the full pipeline: bronze -> silver -> gold -> train
# MAGIC
# MAGIC Set the `api_key` widget below, then Run All. This chains the four
# MAGIC notebooks in order on this cluster - useful for community-edition
# MAGIC workspaces where the scheduled Jobs product isn't available, so a
# MAGIC daily refresh means re-running this notebook (manually, or via the
# MAGIC REST API `jobs/runs/submit` against an existing all-purpose cluster,
# MAGIC if your workspace tier allows it).

# COMMAND ----------

dbutils.widgets.text("api_key", "", "football-data.org API key")

# COMMAND ----------

# MAGIC %run ./01_bronze_ingest

# COMMAND ----------

# MAGIC %run ./02_silver_transform

# COMMAND ----------

# MAGIC %run ./03_gold_features

# COMMAND ----------

# MAGIC %run ./04_train_model
