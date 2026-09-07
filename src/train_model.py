"""
Phase 2c - Baseline model training for the PL Match Predictor.

Trains a LightGBM multiclass classifier (home win / draw / away win) on
the gold feature table, using a TIME-BASED train/val/test split (never
random - random splits leak future form into the past). Uses a
chronological validation slice for early stopping, since the training
set here is small (a few hundred matches) and an untuned model overfits
fast. Every run is logged to MLflow: params, features used, and
accuracy/logloss.
"""

import os
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

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
]

TEST_FRACTION = 0.2   # most recent 20% of finished matches -> held-out test
VAL_FRACTION = 0.15   # next most recent 15% of the remainder -> early-stopping val


def baseline_log_loss(y_test, n_classes):
    """Log loss of a model that just predicts the training class frequencies."""
    freqs = np.bincount(y_test, minlength=n_classes) / len(y_test)
    probs = np.tile(freqs, (len(y_test), 1))
    return log_loss(y_test, probs, labels=list(range(n_classes)))


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

    n = len(df)
    test_idx = int(n * (1 - TEST_FRACTION))
    val_idx = int(test_idx * (1 - VAL_FRACTION))

    X_train, y_train = X.iloc[:val_idx], y[:val_idx]
    X_val, y_val = X.iloc[val_idx:test_idx], y[val_idx:test_idx]
    X_test, y_test = X.iloc[test_idx:], y[test_idx:]

    print(f"Time-based split: {len(X_train)} train / {len(X_val)} val (early stopping) / "
          f"{len(X_test)} test (chronological order: train -> val -> test)")

    mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
    mlflow.set_experiment("pl_match_predictor")
    with mlflow.start_run():
        params = {
            "objective": "multiclass",
            "num_class": len(le.classes_),
            "learning_rate": 0.03,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 20,
            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
            "feature_fraction": 0.8,
            "verbose": -1,
        }
        mlflow.log_params(params)
        mlflow.log_param("features", PRE_MATCH_FEATURES)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("n_test", len(X_test))

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        model = lgb.train(
            params,
            train_set,
            num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(stopping_rounds=25, verbose=False)],
        )

        preds = model.predict(X_test, num_iteration=model.best_iteration)
        pred_labels = preds.argmax(axis=1)

        acc = accuracy_score(y_test, pred_labels)
        ll = log_loss(y_test, preds, labels=list(range(len(le.classes_))))
        baseline_ll = baseline_log_loss(y_test, len(le.classes_))
        majority_acc = pd.Series(y_test).value_counts(normalize=True).max()

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("log_loss", ll)
        mlflow.log_metric("baseline_log_loss", baseline_ll)
        mlflow.log_metric("majority_class_accuracy", majority_acc)
        mlflow.log_metric("best_iteration", model.best_iteration)

        print(f"Best iteration (early stopped): {model.best_iteration}")
        print(f"Test accuracy: {acc:.3f}  (majority-class baseline: {majority_acc:.3f})")
        print(f"Test log loss: {ll:.3f}  (class-frequency baseline: {baseline_ll:.3f})")
        print(f"Classes: {list(le.classes_)}")

        if ll >= baseline_ll:
            print("WARNING: model log loss is not beating the naive class-frequency "
                  "baseline. Treat predictions as unreliable until more finished "
                  "matches accumulate and this is re-checked.")
        else:
            print("Model beats the naive baseline on held-out, time-ordered data.")

        MODEL_DIR.mkdir(exist_ok=True)
        model.save_model(str(MODEL_DIR / "lgbm_model.txt"), num_iteration=model.best_iteration)
        import joblib
        joblib.dump(le, MODEL_DIR / "label_encoder.joblib")
        mlflow.log_artifact(str(MODEL_DIR / "lgbm_model.txt"))

        print(f"Model saved to {MODEL_DIR / 'lgbm_model.txt'}")


if __name__ == "__main__":
    main()
