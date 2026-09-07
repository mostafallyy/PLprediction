"""
Phase 3 - Prediction service for the PL Match Predictor.

Loads the latest gold features and the trained LightGBM model, and
serves win/draw/loss probabilities for upcoming (SCHEDULED) fixtures.

Run locally:
    uvicorn src.app:app --reload --port 8000
Then visit http://localhost:8000/predictions
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException

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

app = FastAPI(title="PL Match Predictor")

_model = None
_label_encoder = None


def load_model():
    global _model, _label_encoder
    model_path = MODEL_DIR / "lgbm_model.txt"
    encoder_path = MODEL_DIR / "label_encoder.joblib"
    if not model_path.exists():
        raise HTTPException(status_code=503, detail="Model not trained yet. Run src/train_model.py first.")
    if _model is None:
        _model = lgb.Booster(model_file=str(model_path))
        _label_encoder = joblib.load(encoder_path)
    return _model, _label_encoder


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predictions")
def predictions():
    """Win/draw/loss probabilities for upcoming (SCHEDULED) fixtures."""
    model, le = load_model()

    gold_path = GOLD_DIR / "match_features.parquet"
    if not gold_path.exists():
        raise HTTPException(status_code=503, detail="No gold feature table yet. Run the pipeline first.")

    df = pd.read_parquet(gold_path)
    upcoming = df[df["status"].isin(["SCHEDULED", "TIMED"])].dropna(subset=PRE_MATCH_FEATURES)

    if upcoming.empty:
        return {"message": "No upcoming fixtures with complete pre-match features yet.", "predictions": []}

    X = upcoming[PRE_MATCH_FEATURES]
    probs = model.predict(X)

    results = []
    for (_, row), p in zip(upcoming.iterrows(), probs):
        prob_by_class = dict(zip(le.classes_, p.tolist()))
        results.append(
            {
                "match_id": int(row["match_id"]),
                "utc_date": str(row["utc_date"]),
                "home_team": row["home_team_name"],
                "away_team": row["away_team_name"],
                "home_win_prob": round(prob_by_class.get("HOME_TEAM", 0.0), 3),
                "draw_prob": round(prob_by_class.get("DRAW", 0.0), 3),
                "away_win_prob": round(prob_by_class.get("AWAY_TEAM", 0.0), 3),
            }
        )

    return {"predictions": results}
