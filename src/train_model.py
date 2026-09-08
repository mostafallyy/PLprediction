"""
Phase 2c - Model training for the PL Match Predictor.

Trains and compares TWO models on the same time-based split:
  - LightGBM (gradient-boosted trees)
  - Logistic Regression (linear baseline, standardized features)

Both are legitimate choices for a small tabular dataset like this one;
rather than assume the more complex model wins, both get trained and
scored on the same held-out, time-ordered test set, against the same
two naive baselines (majority-class, class-frequency). Whichever model
has the LOWER test log loss is saved as the "active" model the API
serves - the choice is made by the numbers, not by default.

Time-based train/val/test split - never random, that would leak future
form into the past. Every run is logged to MLflow.
"""

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import json
import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / "data" / "gold"
MODEL_DIR = ROOT / "models"

PRE_MATCH_FEATURES = [
    "home_form_ppg",
    "home_form_gf",
    "home_form_ga",
    "away_form_ppg",
    "away_form_gf",
    "away_form_ga",
    "home_home_ppg",
    "away_away_ppg",
    "h2h_home_win_rate",
    "home_rest_days",
    "away_rest_days",
    "home_manager_tenure_days",
    "away_manager_tenure_days",
]

TEST_FRACTION = 0.2   # most recent 20% of finished matches -> held-out test
VAL_FRACTION = 0.15   # next most recent 15% of the remainder -> early-stopping val


def baseline_log_loss(y_test, n_classes):
    """Log loss of a model that just predicts the training class frequencies."""
    freqs = np.bincount(y_test, minlength=n_classes) / len(y_test)
    probs = np.tile(freqs, (len(y_test), 1))
    return log_loss(y_test, probs, labels=list(range(n_classes)))


def train_lightgbm(X_train, y_train, X_val, y_val, n_classes):
    params = {
        "objective": "multiclass", "num_class": n_classes,
        "learning_rate": 0.03, "num_leaves": 7, "max_depth": 3,
        "min_data_in_leaf": 20, "lambda_l1": 1.0, "lambda_l2": 1.0,
        "feature_fraction": 0.8, "verbose": -1,
    }
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    model = lgb.train(
        params, train_set, num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
    )
    return model, params, model.best_iteration


def train_logistic_regression(X_train, y_train, n_classes):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    params = {"C": 1.0, "penalty": "l2", "solver": "lbfgs", "max_iter": 2000, "multi_class": "multinomial"}
    model = LogisticRegression(**params)
    model.fit(X_train_scaled, y_train)
    return model, scaler, params


def main():
    gold_path = GOLD_DIR / "match_features.parquet"
    if not gold_path.exists():
        raise SystemExit("No gold table found. Run src/gold_features.py first.")

    df = pd.read_parquet(gold_path)
    df = df[df["status"] == "FINISHED"].dropna(subset=PRE_MATCH_FEATURES + ["winner"])
    df = df.sort_values("utc_date").reset_index(drop=True)

    if len(df) < 20:
        raise SystemExit(
            f"Only {len(df)} finished, feature-complete matches available - "
            "too few for a meaningful time-based split yet."
        )

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

    print(f"Time-based split: {len(X_train)} train / {len(X_val)} val (early stopping) / "
          f"{len(X_test)} test (chronological order: train -> val -> test)")

    baseline_ll = baseline_log_loss(y_test, n_classes)
    majority_acc = pd.Series(y_test).value_counts(normalize=True).max()

    mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
    mlflow.set_experiment("pl_match_predictor")

    results = {}

    # --- LightGBM ---
    with mlflow.start_run(run_name="lightgbm"):
        lgbm_model, lgbm_params, best_iter = train_lightgbm(X_train, y_train, X_val, y_val, n_classes)
        preds = lgbm_model.predict(X_test, num_iteration=best_iter)
        acc = accuracy_score(y_test, preds.argmax(axis=1))
        ll = log_loss(y_test, preds, labels=list(range(n_classes)))
        mlflow.log_params({**lgbm_params, "best_iteration": best_iter})
        mlflow.log_metrics({"accuracy": acc, "log_loss": ll, "baseline_log_loss": baseline_ll, "majority_class_accuracy": majority_acc})
        results["lightgbm"] = {"accuracy": acc, "log_loss": ll, "model": lgbm_model, "best_iteration": best_iter}
        print(f"[LightGBM]            accuracy {acc:.3f}   log loss {ll:.3f}   (best_iter={best_iter})")

    # --- Logistic Regression ---
    with mlflow.start_run(run_name="logistic_regression"):
        logreg_model, scaler, logreg_params = train_logistic_regression(X_train, y_train, n_classes)
        preds = logreg_model.predict_proba(scaler.transform(X_test))
        # sklearn orders predict_proba columns by logreg_model.classes_, which
        # matches the LabelEncoder's 0..n-1 ordering since we fit on the same y
        acc = accuracy_score(y_test, preds.argmax(axis=1))
        ll = log_loss(y_test, preds, labels=list(range(n_classes)))
        mlflow.log_params(logreg_params)
        mlflow.log_metrics({"accuracy": acc, "log_loss": ll, "baseline_log_loss": baseline_ll, "majority_class_accuracy": majority_acc})
        results["logistic_regression"] = {"accuracy": acc, "log_loss": ll, "model": logreg_model, "scaler": scaler}
        print(f"[Logistic Regression] accuracy {acc:.3f}   log loss {ll:.3f}")

    print(f"[Naive baselines]     majority-class accuracy {majority_acc:.3f}   class-frequency log loss {baseline_ll:.3f}")

    # --- pick the winner by test log loss (proper scoring rule - rewards calibrated
    # probabilities, not just the argmax being right) ---
    winner = min(results, key=lambda k: results[k]["log_loss"])
    print(f"\nSelected model: {winner} (lower test log loss)")

    MODEL_DIR.mkdir(exist_ok=True)

    if winner == "lightgbm":
        m = results["lightgbm"]["model"]
        m.save_model(str(MODEL_DIR / "lgbm_model.txt"), num_iteration=results["lightgbm"]["best_iteration"])
    else:
        joblib.dump(results["logistic_regression"]["model"], MODEL_DIR / "logreg_model.joblib")
        joblib.dump(results["logistic_regression"]["scaler"], MODEL_DIR / "logreg_scaler.joblib")

    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")

    active_model = {
        "active_model_type": winner,
        "features": PRE_MATCH_FEATURES,
        "comparison": {
            k: {"accuracy": round(v["accuracy"], 4), "log_loss": round(v["log_loss"], 4)}
            for k, v in results.items()
        },
        "naive_baselines": {
            "majority_class_accuracy": round(float(majority_acc), 4),
            "class_frequency_log_loss": round(float(baseline_ll), 4),
        },
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    (MODEL_DIR / "active_model.json").write_text(json.dumps(active_model, indent=2))

    for name, r in results.items():
        tag = " <- ACTIVE" if name == winner else ""
        beats = "beats" if r["log_loss"] < baseline_ll else "does NOT beat"
        print(f"  {name}: {beats} the naive baseline on held-out data{tag}")

    print(f"\nSaved active model ({winner}) + comparison metadata to {MODEL_DIR}/active_model.json")


if __name__ == "__main__":
    main()
