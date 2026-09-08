"""
One-off export: re-runs the exact same fitting logic as
classify_players_logreg.py but dumps every number needed for the
explainer artifact (confusion matrices, ROC curve points, per-class
AUC, classification report, statsmodels coefficient tables) to a
single JSON file instead of printing to stdout.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify_players_logreg import GROUP_FEATURES, TIER_LABELS, build_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "external_data" / "player_season_stats.csv"
OUT_JSON = ROOT / "models" / "player_tier_logreg" / "report_data.json"


def main():
    raw = pd.read_csv(DATA_PATH)
    df = build_features(raw)

    groups_out = {}
    for group, features in GROUP_FEATURES.items():
        gdf = df[df["primary_pos"] == group].dropna(subset=features + ["tier"]).copy()
        if len(gdf) < 50:
            continue

        X = gdf[features].values
        y = gdf["tier"].values
        pgroups = gdf["player_id"].values

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, pgroups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(X_train_s, y_train)
        proba = clf.predict_proba(X_test_s)
        preds = clf.classes_[proba.argmax(axis=1)]

        acc = accuracy_score(y_test, preds)
        report = classification_report(
            y_test, preds, labels=clf.classes_,
            target_names=[TIER_LABELS[c] for c in clf.classes_], output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(y_test, preds, labels=clf.classes_).tolist()

        y_test_bin = label_binarize(y_test, classes=clf.classes_)
        per_class_auc = {}
        roc_curves = {}
        for i, c in enumerate(clf.classes_):
            if y_test_bin[:, i].sum() == 0 or y_test_bin[:, i].sum() == len(y_test_bin):
                continue
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba[:, i])
            auc = roc_auc_score(y_test_bin[:, i], proba[:, i])
            per_class_auc[TIER_LABELS[c]] = round(float(auc), 4)
            # downsample if very long, keep endpoints
            idx = np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int)
            roc_curves[TIER_LABELS[c]] = {
                "fpr": [round(float(v), 4) for v in fpr[idx]],
                "tpr": [round(float(v), 4) for v in tpr[idx]],
            }
        macro_auc = float(np.mean(list(per_class_auc.values()))) if per_class_auc else None

        # statsmodels inference on the SAME training rows
        y_recoded = 5 - y_train
        X_sm = sm.add_constant(pd.DataFrame(X_train_s, columns=features).reset_index(drop=True))
        y_sm = pd.Series(y_recoded).reset_index(drop=True)
        coefficients = []
        if y_sm.nunique() >= 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    mn_result = sm.MNLogit(y_sm, X_sm).fit(method="newton", maxiter=200, disp=0)
                    recoded_to_tier = {0: 4, 1: 3, 2: 2, 3: 1}
                    for col in mn_result.params.columns:
                        tier = recoded_to_tier.get(col, col)
                        for feat in mn_result.params.index:
                            c = float(mn_result.params.loc[feat, col])
                            se = float(mn_result.bse.loc[feat, col])
                            z = float(mn_result.tvalues.loc[feat, col])
                            p = float(mn_result.pvalues.loc[feat, col])
                            coefficients.append({
                                "tier": tier, "tier_label": TIER_LABELS[tier],
                                "feature": feat, "coef": round(c, 3), "se": round(se, 3),
                                "z": round(z, 2), "p": round(p, 4),
                                "significant": bool(p < 0.05),
                            })
                except Exception as e:
                    print(f"  [{group}] MNLogit failed: {e}")

        groups_out[group] = {
            "n": len(gdf), "n_train": len(X_train), "n_test": len(X_test),
            "features": features, "accuracy": round(float(acc), 4),
            "macro_auc": round(macro_auc, 4) if macro_auc else None,
            "classes": [TIER_LABELS[c] for c in clf.classes_],
            "classification_report": report,
            "confusion_matrix": cm,
            "per_class_auc": per_class_auc,
            "roc_curves": roc_curves,
            "coefficients": coefficients,
        }
        print(f"{group}: exported ({len(gdf)} rows, acc={acc:.3f}, macro_auc={macro_auc:.3f})")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(groups_out, indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
