"""
Phase 2f - Logistic regression player-tier classifier, WITH full
statistical inference (coefficients, z, p, 95% CI) and ROC/AUC.

Replaces the hand-picked weights in ingest_player_stats.py's
classify_impact() (attack_weight/defense_weight tables, the 0.5
multiplier) with weights a model actually LEARNS from the data. The
TARGET is still the same 5-tier label that formula assigns (Talisman
... Fringe/Backup) - so this is explicitly "can a model reproduce, and
explain the drivers behind, the tier a rule-based formula already
assigns" rather than an independent ground truth. That's a deliberate
choice, not an oversight: it turns a hand-tuned formula into
data-driven, statistically testable weights per position, the same way
train_model.py replaced "which model do I assume is better" with "let
the held-out test log loss decide."

ONE MODEL PER POSITION GROUP (FW / MF / DF / GK), not one shared model,
because what makes a forward valuable and what makes a defender valuable
are genuinely different questions with different relevant stats - a
single model would need irrelevant zero-filled columns for 3 out of 4
groups on every row. Position bucketed by FBref's own convention (the
FIRST 2-letter code in a multi-position tag is primary - "FWMF" is a
forward listed as also playing midfield).

Features per group (only stats we actually have - no key passes, no
blocks, no pass-completion %, since those live in FBref tables we
haven't pulled; noted plainly rather than faked):
  - FW: gls_per90, ast_per90, gapk_per90 (attacking output)
  - MF: gapk_per90, def_actions_per90 (blend of attack + defense -
        this group is the closest we can get to "well-rounded"
        without passing/progression data)
  - DF: def_actions_per90, team_cs_pct (the squad's primary
        goalkeeper's clean-sheet% that season - individual defenders
        aren't attributed personal clean sheets in this data, so a
        squad-level defensive-solidity proxy is used instead, shared
        by every defender on that squad-season)
  - GK: save_pct, cs_pct, saves_per90, ga90 (lower is better - sign
        flipped implicitly by the model's own coefficient)
  - EVERY group also gets minutes_share ("bonus for minutes played",
    per spec) - how much of the squad's total minutes this player got.

TWO fitting passes, on purpose, because they answer different
questions:
  1. sklearn LogisticRegression (multinomial/softmax) - fit on a
     GROUP-SHUFFLE train/test split (split by player_id, not by row,
     so the SAME PLAYER's different seasons never land in both train
     and test - otherwise the model could partly "memorize" a
     player's typical profile instead of learning general patterns).
     Scored on the held-out test set: accuracy, per-class precision/
     recall/F1, confusion matrix, one-vs-rest ROC curves + AUC per
     tier (multi-class AUC isn't a single well-defined number the way
     binary AUC is - one-vs-rest treats "is this Tier 1 vs not" as its
     own binary problem per class, then macro-averages).
  2. statsmodels MNLogit (multinomial logit) on the SAME TRAINING rows
     - not test rows, so the held-out set stays clean for the sklearn
     model's numbers - purely for INFERENCE: coefficient estimate,
     standard error, z-statistic, p-value (P>|z|), and a 95% CI per
     feature per tier (relative to Tier 5/Fringe-Backup, the reference
     category statsmodels picks by default = the lowest-numbered
     class... note MNLogit actually uses the FIRST category in sorted
     order as reference, which here is Tier 1 - handled explicitly
     below by re-coding so Tier 5 is the reference, since "how much
     more likely than being a fringe player" is the more interpretable
     framing). A small p-value (conventionally <0.05) means we can be
     fairly confident that feature's relationship with tier isn't just
     noise; the z-statistic is the coefficient divided by its standard
     error - larger |z| = more confident the true effect isn't zero.
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, label_binarize

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "external_data" / "player_season_stats.csv"
OUT_DIR = ROOT / "models" / "player_tier_logreg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIER_LABELS = {1: "Talisman", 2: "Key Player", 3: "Regular Starter", 4: "Squad Rotation", 5: "Fringe/Backup"}

GROUP_FEATURES = {
    "FW": ["gls_per90", "ast_per90", "gapk_per90", "minutes_share"],
    "MF": ["gapk_per90", "def_actions_per90", "minutes_share"],
    "DF": ["def_actions_per90", "team_cs_pct", "minutes_share"],
    "GK": ["save_pct", "cs_pct", "saves_per90", "ga90", "minutes_share"],
}


def primary_position(pos: str) -> str:
    pos = str(pos)
    for code in ("GK", "FW", "MF", "DF"):
        if pos.startswith(code):
            return code
    return "UNK"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["primary_pos"] = df["pos"].apply(primary_position)

    squad_total_nineties = df.groupby(["squad_key", "season_start_year"])["nineties"].transform("sum")
    df["minutes_share"] = (df["nineties"] / squad_total_nineties).fillna(0)

    df["saves_per90"] = (df["saves"] / df["nineties"].replace(0, np.nan)).fillna(0)

    # team_cs_pct: the squad's PRIMARY goalkeeper's (most minutes played)
    # clean-sheet % that season, broadcast to every player on the squad -
    # the only clean-sheet signal available for outfield players, since
    # individual defenders aren't attributed personal clean sheets here.
    gk_rows = df[df["primary_pos"] == "GK"].sort_values("nineties", ascending=False)
    primary_gk_cs = gk_rows.drop_duplicates(subset=["squad_key", "season_start_year"])[
        ["squad_key", "season_start_year", "cs_pct"]
    ].rename(columns={"cs_pct": "team_cs_pct"})
    df = df.merge(primary_gk_cs, on=["squad_key", "season_start_year"], how="left")

    return df


def fit_sklearn_model(X_train, y_train, X_test, y_test, feature_names, group_name):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=3000, C=1.0)
    clf.fit(X_train_s, y_train)

    proba = clf.predict_proba(X_test_s)
    preds = clf.classes_[proba.argmax(axis=1)]

    acc = accuracy_score(y_test, preds)
    report = classification_report(
        y_test, preds, labels=clf.classes_,
        target_names=[TIER_LABELS[c] for c in clf.classes_], digits=3, zero_division=0,
    )
    cm = confusion_matrix(y_test, preds, labels=clf.classes_)

    # one-vs-rest ROC/AUC per class - only defined for classes present in
    # BOTH train and test with >1 example, guard for tiny groups (GK).
    y_test_bin = label_binarize(y_test, classes=clf.classes_)
    aucs = {}
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, c in enumerate(clf.classes_):
        if y_test_bin[:, i].sum() == 0 or y_test_bin[:, i].sum() == len(y_test_bin):
            continue  # class absent (or universal) in this test fold - AUC undefined
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba[:, i])
        auc = roc_auc_score(y_test_bin[:, i], proba[:, i])
        aucs[TIER_LABELS[c]] = auc
        ax.plot(fpr, tpr, label=f"{TIER_LABELS[c]} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{group_name}: one-vs-rest ROC curves (test set)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"roc_{group_name}.png", dpi=130)
    plt.close(fig)

    macro_auc = np.mean(list(aucs.values())) if aucs else float("nan")

    cm_fig, cm_ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=[TIER_LABELS[c] for c in clf.classes_]).plot(
        ax=cm_ax, cmap="Purples", colorbar=False, xticks_rotation=45
    )
    cm_ax.set_title(f"{group_name}: confusion matrix (test set)")
    cm_fig.tight_layout()
    cm_fig.savefig(OUT_DIR / f"confusion_{group_name}.png", dpi=130)
    plt.close(cm_fig)

    return {
        "accuracy": acc, "report": report, "confusion_matrix": cm,
        "classes": clf.classes_, "per_class_auc": aucs, "macro_auc": macro_auc,
        "coefficients": dict(zip(feature_names, clf.coef_.tolist())) if len(clf.classes_) == 2 else None,
    }


def fit_statsmodels_inference(X_train, y_train, feature_names, group_name):
    """Multinomial logit for coefficient significance - fit on the SAME
    training rows as the sklearn model (test set stays untouched)."""
    # statsmodels MNLogit uses the numerically-lowest class as reference.
    # Re-code so Tier 5 (Fringe/Backup) is reference (coded 0) - "how much
    # more likely than being a fringe player" is the readable framing.
    y_recoded = 5 - y_train  # tier 5 -> 0 (reference), tier 1 -> 4, etc.

    X_sm = sm.add_constant(pd.DataFrame(X_train, columns=feature_names).reset_index(drop=True))
    y_sm = pd.Series(y_recoded).reset_index(drop=True)

    if y_sm.nunique() < 2:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = sm.MNLogit(y_sm, X_sm)
            result = model.fit(method="newton", maxiter=200, disp=0)
        except Exception as e:  # perfect separation / singular matrix on a tiny group
            print(f"  [{group_name}] statsmodels MNLogit failed to converge: {e}")
            return None

    return result


def print_inference_table(result, feature_names, group_name):
    if result is None:
        print(f"  (statsmodels inference unavailable for {group_name} - see warning above)")
        return
    params = result.params
    bse = result.bse
    zvals = result.tvalues
    pvals = result.pvalues

    # columns are the non-reference outcome categories, in the recoded
    # order (0=Tier5 reference, so columns are 1,2,3,4 -> Tier4,3,2,1)
    recoded_to_tier = {0: 4, 1: 3, 2: 2, 3: 1}
    for col in params.columns:
        tier = recoded_to_tier.get(col, col)
        print(f"\n  --- {group_name}: log-odds of Tier {tier} ({TIER_LABELS.get(tier,'?')}) vs Tier 5 (Fringe/Backup) ---")
        print(f"  {'feature':<18}{'coef':>10}{'std err':>10}{'z':>9}{'P>|z|':>9}{'95% CI':>18}  sig")
        for feat in params.index:
            c = params.loc[feat, col]
            se = bse.loc[feat, col]
            z = zvals.loc[feat, col]
            p = pvals.loc[feat, col]
            lo, hi = c - 1.96 * se, c + 1.96 * se
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {feat:<18}{c:>10.3f}{se:>10.3f}{z:>9.2f}{p:>9.3f}   [{lo:6.2f}, {hi:6.2f}]  {sig}")


def main():
    if not DATA_PATH.exists():
        sys.exit(f"{DATA_PATH} not found - run src/ingest_player_stats.py first.")

    raw = pd.read_csv(DATA_PATH)
    df = build_features(raw)

    print(f"Loaded {len(df)} player-seasons. Position groups: "
          f"{df['primary_pos'].value_counts().to_dict()}\n")

    summary = {}
    for group, features in GROUP_FEATURES.items():
        gdf = df[df["primary_pos"] == group].dropna(subset=features + ["tier"]).copy()
        if len(gdf) < 50:
            print(f"=== {group}: only {len(gdf)} usable rows - skipping ===\n")
            continue

        X = gdf[features].values
        y = gdf["tier"].values
        groups = gdf["player_id"].values

        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"=== {group} ({len(gdf)} player-seasons; {len(X_train)} train / {len(X_test)} test, "
              f"split by player so no player appears in both) ===")
        print(f"Features: {features}")

        sk_result = fit_sklearn_model(X_train, y_train, X_test, y_test, features, group)
        print(f"\nAccuracy: {sk_result['accuracy']:.3f}   Macro-average one-vs-rest AUC: {sk_result['macro_auc']:.3f}")
        print(sk_result["report"])
        print(f"Per-class AUC: {sk_result['per_class_auc']}")
        print(f"ROC curve saved -> {OUT_DIR / f'roc_{group}.png'}")
        print(f"Confusion matrix saved -> {OUT_DIR / f'confusion_{group}.png'}")

        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        sm_result = fit_statsmodels_inference(X_train_s, y_train, features, group)
        print_inference_table(sm_result, features, group)

        summary[group] = {
            "n": len(gdf), "accuracy": sk_result["accuracy"], "macro_auc": sk_result["macro_auc"],
        }
        print("\n" + "=" * 90 + "\n")

    print("Summary across position groups:")
    for g, s in summary.items():
        print(f"  {g}: n={s['n']:5d}  accuracy={s['accuracy']:.3f}  macro AUC={s['macro_auc']:.3f}")


if __name__ == "__main__":
    main()
