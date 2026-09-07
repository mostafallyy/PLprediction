# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2c - Model training (LightGBM, Databricks-managed MLflow)
# MAGIC
# MAGIC LightGBM isn't a distributed Spark algorithm, so the gold table -
# MAGIC a few hundred rows - is pulled to the driver with `toPandas()` and
# MAGIC trained there. That's standard practice for small tabular models
# MAGIC on Databricks; the point of doing the FEATURE work in Spark is that
# MAGIC it's what actually needs to scale (window functions over the full
# MAGIC match history), not the final fit.
# MAGIC
# MAGIC Time-based train/val/test split - never random, that would leak
# MAGIC future form into the past. Every run is logged to the notebook's
# MAGIC attached MLflow experiment automatically.

# COMMAND ----------

# MAGIC %pip install lightgbm

# COMMAND ----------

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

GOLD_DB = "pl_predictor_gold"

PRE_MATCH_FEATURES = [
    "home_form_ppg", "home_form_gf", "home_form_ga",
    "away_form_ppg", "away_form_gf", "away_form_ga",
    "home_home_ppg", "away_away_ppg",
    "h2h_home_win_rate", "home_rest_days", "away_rest_days",
]
TEST_FRACTION = 0.2
VAL_FRACTION = 0.15

# COMMAND ----------

gold_sdf = spark.table(f"{GOLD_DB}.match_features")
df = (
    gold_sdf.filter("status = 'FINISHED'")
    .dropna(subset=PRE_MATCH_FEATURES + ["winner"])
    .orderBy("utc_date")
    .toPandas()
)

if len(df) < 20:
    raise ValueError(f"Only {len(df)} finished, feature-complete matches - run 01-03 first / wait for more results.")

le = LabelEncoder()
y = le.fit_transform(df["winner"])
X = df[PRE_MATCH_FEATURES]

n = len(df)
test_idx = int(n * (1 - TEST_FRACTION))
val_idx = int(test_idx * (1 - VAL_FRACTION))

X_train, y_train = X.iloc[:val_idx], y[:val_idx]
X_val, y_val = X.iloc[val_idx:test_idx], y[val_idx:test_idx]
X_test, y_test = X.iloc[test_idx:], y[test_idx:]

print(f"Time-based split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

# COMMAND ----------

def baseline_log_loss(y_test, n_classes):
    freqs = np.bincount(y_test, minlength=n_classes) / len(y_test)
    probs = np.tile(freqs, (len(y_test), 1))
    return log_loss(y_test, probs, labels=list(range(n_classes)))


with mlflow.start_run() as run:
    params = {
        "objective": "multiclass", "num_class": len(le.classes_),
        "learning_rate": 0.03, "num_leaves": 7, "max_depth": 3,
        "min_data_in_leaf": 20, "lambda_l1": 1.0, "lambda_l2": 1.0,
        "feature_fraction": 0.8, "verbose": -1,
    }
    mlflow.log_params(params)
    mlflow.log_param("features", PRE_MATCH_FEATURES)
    mlflow.log_param("n_train", len(X_train))
    mlflow.log_param("n_val", len(X_val))
    mlflow.log_param("n_test", len(X_test))

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    model = lgb.train(
        params, train_set, num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
    )

    preds = model.predict(X_test, num_iteration=model.best_iteration)
    acc = accuracy_score(y_test, preds.argmax(axis=1))
    ll = log_loss(y_test, preds, labels=list(range(len(le.classes_))))
    baseline_ll = baseline_log_loss(y_test, len(le.classes_))
    majority_acc = pd.Series(y_test).value_counts(normalize=True).max()

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("log_loss", ll)
    mlflow.log_metric("baseline_log_loss", baseline_ll)
    mlflow.log_metric("majority_class_accuracy", majority_acc)
    mlflow.log_metric("best_iteration", model.best_iteration)

    print(f"Test accuracy: {acc:.3f} (majority baseline: {majority_acc:.3f})")
    print(f"Test log loss: {ll:.3f} (class-frequency baseline: {baseline_ll:.3f})")
    print("Beats naive baseline." if ll < baseline_ll else "WARNING: does not beat naive baseline.")

    model_path = "/tmp/lgbm_model.txt"
    model.save_model(model_path, num_iteration=model.best_iteration)
    mlflow.log_artifact(model_path)

    dbutils.fs.cp(f"file:{model_path}", "dbfs:/pl_predictor/models/lgbm_model.txt", recurse=False)
    print(f"MLflow run: {run.info.run_id}")
    print("Model copied to dbfs:/pl_predictor/models/lgbm_model.txt")
