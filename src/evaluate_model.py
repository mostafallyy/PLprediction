"""
Phase 2c-eval - Detailed classification metrics for the PL Match Predictor.

train_model.py only ever logs accuracy + log loss (the metric the winner
is actually chosen by - log loss is a proper scoring rule, accuracy isn't).
This script reproduces the EXACT same time-based split and re-scores both
models on the held-out test set with sklearn's full classification_report
(per-class precision/recall/f1) plus a confusion matrix, so the numbers
below are apples-to-apples with what train_model.py already reported -
nothing re-trained, nothing re-split.
"""

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gold_features import CORE_FEATURES, ENRICHMENT_FEATURES, PRE_MATCH_FEATURES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / "data" / "gold"
MODEL_DIR = ROOT / "models"

TEST_FRACTION = 0.2
VAL_FRACTION = 0.15


def main():
    df = pd.read_parquet(GOLD_DIR / "match_features.parquet")
    df = df[df["status"] == "FINISHED"].dropna(subset=CORE_FEATURES + ["winner"])
    df = df.sort_values("utc_date").reset_index(drop=True)

    le = joblib.load(MODEL_DIR / "label_encoder.joblib")
    y = le.transform(df["winner"])
    X = df[PRE_MATCH_FEATURES].copy()

    n = len(df)
    test_idx = int(n * (1 - TEST_FRACTION))
    val_idx = int(test_idx * (1 - VAL_FRACTION))

    X_train = X.iloc[:val_idx].copy()
    X_test, y_test = X.iloc[test_idx:].copy(), y[test_idx:]

    impute_medians = {c: float(X_train[c].median()) for c in ENRICHMENT_FEATURES}
    for c in ENRICHMENT_FEATURES:
        X_test[c] = X_test[c].fillna(impute_medians[c])

    print(f"Test set: {len(X_test)} matches (most recent {TEST_FRACTION:.0%} chronologically, "
          f"never seen during training or early-stopping)")
    print(f"Classes: {list(le.classes_)}\n")

    def report(name, y_pred, y_proba=None):
        print(f"=== {name} ===")
        print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3, zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(le.classes_))))
        print("Confusion matrix (rows = actual, cols = predicted):")
        header = "".ljust(14) + "".join(c.ljust(14) for c in le.classes_)
        print(header)
        for i, row in enumerate(cm):
            print(le.classes_[i].ljust(14) + "".join(str(v).ljust(14) for v in row))
        print()

    # Only the WINNING model from the most recent train_model.py run gets
    # its artifact re-saved to disk each time - the loser'''s old file can be
    # stale (fit on an earlier, shorter feature list) and isn'''t safe to
    # score against today'''s feature set. Evaluate whichever is active.
    import json
    active = json.loads((MODEL_DIR / "active_model.json").read_text())["active_model_type"]

    if active == "lightgbm" and (MODEL_DIR / "lgbm_model.txt").exists():
        lgbm = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_model.txt"))
        proba = lgbm.predict(X_test)
        preds = proba.argmax(axis=1)
        report("LightGBM (ACTIVE - currently served)", preds, proba)
    elif active == "logistic_regression" and (MODEL_DIR / "logreg_model.joblib").exists():
        logreg = joblib.load(MODEL_DIR / "logreg_model.joblib")
        scaler = joblib.load(MODEL_DIR / "logreg_scaler.joblib")
        proba = logreg.predict_proba(scaler.transform(X_test))
        preds = proba.argmax(axis=1)
        report("Logistic Regression (ACTIVE - currently served)", preds, proba)

    # --- naive baselines, for context ---
    majority_class = pd.Series(y_test).mode()[0]
    majority_preds = np.full_like(y_test, majority_class)
    report(f"Naive baseline (always predict '{le.classes_[majority_class]}')", majority_preds)


if __name__ == "__main__":
    main()
