# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2c - Model training (Databricks-managed MLflow)
# MAGIC
# MAGIC Trains and compares TWO models on the same time-based split, same
# MAGIC as the local pipeline (src/train_model.py):
# MAGIC   - LightGBM (gradient-boosted trees)
# MAGIC   - Logistic Regression (linear baseline, standardized features)
# MAGIC
# MAGIC Neither is a distributed Spark algorithm, so the gold table - a few
# MAGIC thousand rows now that Kaggle history is merged in - is pulled to the
# MAGIC driver with `toPandas()` and trained there. That's standard practice
# MAGIC for small tabular models on Databricks; the point of doing the
# MAGIC FEATURE work in Spark is that it's what actually needs to scale
# MAGIC (window functions over the full match history), not the final fit.
# MAGIC
# MAGIC Whichever model has the LOWER test log loss (a proper scoring rule -
# MAGIC rewards calibrated probabilities, not just the argmax being right) is
# MAGIC published as the active model. Time-based train/val/test split -
# MAGIC never random, that would leak future form into the past. Both runs
# MAGIC are logged to the notebook's attached MLflow experiment.

# COMMAND ----------

# MAGIC %pip install lightgbm

# COMMAND ----------

import base64
import json as _json
import time as _time

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, LongType, DoubleType
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler

GOLD_DB = "pl_predictor_gold"

PRE_MATCH_FEATURES = [
    "home_form_ppg", "home_form_gf", "home_form_ga",
    "away_form_ppg", "away_form_gf", "away_form_ga",
    "home_home_ppg", "away_away_ppg",
    "h2h_home_win_rate", "home_rest_days", "away_rest_days",
    "home_manager_tenure_days", "away_manager_tenure_days",
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
n_classes = len(le.classes_)

n = len(df)
test_idx = int(n * (1 - TEST_FRACTION))
val_idx = int(test_idx * (1 - VAL_FRACTION))

X_train, y_train = X.iloc[:val_idx], y[:val_idx]
X_val, y_val = X.iloc[val_idx:test_idx], y[val_idx:test_idx]
X_test, y_test = X.iloc[test_idx:], y[test_idx:]

print(f"Time-based split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test "
      f"(chronological order: train -> val -> test)")

# COMMAND ----------

def baseline_log_loss(y_test, n_classes):
    freqs = np.bincount(y_test, minlength=n_classes) / len(y_test)
    probs = np.tile(freqs, (len(y_test), 1))
    return log_loss(y_test, probs, labels=list(range(n_classes)))


baseline_ll = baseline_log_loss(y_test, n_classes)
majority_acc = pd.Series(y_test).value_counts(normalize=True).max()

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks")

results = {}

# --- LightGBM ---
with mlflow.start_run(run_name="lightgbm") as run:
    params = {
        "objective": "multiclass", "num_class": n_classes,
        "learning_rate": 0.03, "num_leaves": 7, "max_depth": 3,
        "min_data_in_leaf": 20, "lambda_l1": 1.0, "lambda_l2": 1.0,
        "feature_fraction": 0.8, "verbose": -1,
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    lgbm_model = lgb.train(
        params, train_set, num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
    )
    preds = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)
    acc = accuracy_score(y_test, preds.argmax(axis=1))
    ll = log_loss(y_test, preds, labels=list(range(n_classes)))

    mlflow.log_params({**params, "best_iteration": lgbm_model.best_iteration})
    mlflow.log_param("features", PRE_MATCH_FEATURES)
    mlflow.log_metrics({"accuracy": acc, "log_loss": ll, "baseline_log_loss": baseline_ll, "majority_class_accuracy": majority_acc})

    results["lightgbm"] = {"accuracy": acc, "log_loss": ll, "model": lgbm_model, "best_iteration": lgbm_model.best_iteration, "run_id": run.info.run_id}
    print(f"[LightGBM]            accuracy {acc:.3f}   log loss {ll:.3f}   (best_iter={lgbm_model.best_iteration})")

# --- Logistic Regression ---
with mlflow.start_run(run_name="logistic_regression") as run:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    logreg_params = {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 2000}
    logreg_model = LogisticRegression(**logreg_params)
    logreg_model.fit(X_train_scaled, y_train)

    preds = logreg_model.predict_proba(scaler.transform(X_test))
    acc = accuracy_score(y_test, preds.argmax(axis=1))
    ll = log_loss(y_test, preds, labels=list(range(n_classes)))

    mlflow.log_params(logreg_params)
    mlflow.log_param("features", PRE_MATCH_FEATURES)
    mlflow.log_metrics({"accuracy": acc, "log_loss": ll, "baseline_log_loss": baseline_ll, "majority_class_accuracy": majority_acc})

    results["logistic_regression"] = {"accuracy": acc, "log_loss": ll, "model": logreg_model, "scaler": scaler, "run_id": run.info.run_id}
    print(f"[Logistic Regression] accuracy {acc:.3f}   log loss {ll:.3f}")

print(f"[Naive baselines]     majority-class accuracy {majority_acc:.3f}   class-frequency log loss {baseline_ll:.3f}")

# --- pick the winner by test log loss ---
winner = min(results, key=lambda k: results[k]["log_loss"])
print(f"\nSelected model: {winner} (lower test log loss)")
for name, r in results.items():
    tag = " <- ACTIVE" if name == winner else ""
    beats = "beats" if r["log_loss"] < baseline_ll else "does NOT beat"
    print(f"  {name}: {beats} the naive baseline on held-out data{tag}")

# COMMAND ----------

# MAGIC %md
# MAGIC Both models are logged as MLflow artifacts on their own runs above
# MAGIC (retrievable via the MLflow API/UI). Unity Catalog blocks direct
# MAGIC DBFS/artifact-proxy downloads on this workspace though, so - same
# MAGIC workaround as before - the WINNING model is also published as
# MAGIC base64 in a Delta table via the SQL Warehouse path, which is the
# MAGIC channel that's actually reachable from outside the workspace.

# COMMAND ----------

model_row = {
    "trained_at": int(_time.time() * 1000),
    "active_model_type": winner,
    "run_id": results[winner]["run_id"],
    "label_classes": _json.dumps(list(le.classes_)),
    "features": _json.dumps(PRE_MATCH_FEATURES),
    "accuracy": float(results[winner]["accuracy"]),
    "log_loss": float(results[winner]["log_loss"]),
    "comparison": _json.dumps({k: {"accuracy": round(v["accuracy"], 4), "log_loss": round(v["log_loss"], 4)} for k, v in results.items()}),
    "naive_baselines": _json.dumps({"majority_class_accuracy": round(float(majority_acc), 4), "class_frequency_log_loss": round(float(baseline_ll), 4)}),
    "lgbm_model_b64": None,
    "logreg_model_b64": None,
    "logreg_scaler_b64": None,
}

if winner == "lightgbm":
    model_path = "/tmp/lgbm_model.txt"
    results["lightgbm"]["model"].save_model(model_path, num_iteration=results["lightgbm"]["best_iteration"])
    with open(model_path, "rb") as f:
        model_row["lgbm_model_b64"] = base64.b64encode(f.read()).decode("ascii")
    mlflow.log_artifact(model_path)
else:
    import joblib
    model_path = "/tmp/logreg_model.joblib"
    scaler_path = "/tmp/logreg_scaler.joblib"
    joblib.dump(results["logistic_regression"]["model"], model_path)
    joblib.dump(results["logistic_regression"]["scaler"], scaler_path)
    with open(model_path, "rb") as f:
        model_row["logreg_model_b64"] = base64.b64encode(f.read()).decode("ascii")
    with open(scaler_path, "rb") as f:
        model_row["logreg_scaler_b64"] = base64.b64encode(f.read()).decode("ascii")

# Explicit schema - model_row always has at least one None column
# (whichever model type did NOT win), and Spark can't infer a type for
# an all-None column via createDataFrame's normal type inference.
model_row_schema = StructType([
    StructField("trained_at", LongType()),
    StructField("active_model_type", StringType()),
    StructField("run_id", StringType()),
    StructField("label_classes", StringType()),
    StructField("features", StringType()),
    StructField("accuracy", DoubleType()),
    StructField("log_loss", DoubleType()),
    StructField("comparison", StringType()),
    StructField("naive_baselines", StringType()),
    StructField("lgbm_model_b64", StringType()),
    StructField("logreg_model_b64", StringType()),
    StructField("logreg_scaler_b64", StringType()),
])

spark.createDataFrame([model_row], schema=model_row_schema).write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(f"{GOLD_DB}.latest_model")

print(f"\nPublished active model ({winner}) + comparison metadata to {GOLD_DB}.latest_model")
