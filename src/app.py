"""
Phase 3 - Prediction service for the PL Match Predictor.

Loads the latest gold features, the trained model, and team/squad data,
and serves:
  - /predictions       win/draw/loss probabilities for ALL upcoming fixtures
  - /matchweeks        list of matchdays with fixture counts + date ranges
  - /matchweek/{n}     fixtures (+ predictions or results) for one matchday
  - /team/{id}         crest, colors, squad roster for one team
  - /h2h/{a_id}/{b_id}  head-to-head match history between two teams
  - /                  the dashboard frontend (static/index.html)

Run locally:
    uvicorn src.app:app --reload --port 8000
Then visit http://localhost:8000/
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
GOLD_DIR = ROOT / "data" / "gold"
MODEL_DIR = ROOT / "models"
STATIC_DIR = ROOT / "static"

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
_scaler = None
_model_type = None
_label_encoder = None
_teams = None


def load_model():
    """Loads whichever model won the train_model.py comparison (LightGBM or
    Logistic Regression), as recorded in models/active_model.json. Both
    model types are trained on the identical feature set and time-based
    split, so either can be "active" - this just needs to know how to
    load and call whichever one it is."""
    global _model, _scaler, _model_type, _label_encoder
    meta_path = MODEL_DIR / "active_model.json"
    encoder_path = MODEL_DIR / "label_encoder.joblib"
    if not meta_path.exists():
        raise HTTPException(status_code=503, detail="Model not trained yet. Run src/train_model.py first.")
    if _model is not None:
        return _model, _label_encoder

    meta = json.loads(meta_path.read_text())
    _model_type = meta["active_model_type"]
    _label_encoder = joblib.load(encoder_path)

    if _model_type == "lightgbm":
        model_path = MODEL_DIR / "lgbm_model.txt"
        if not model_path.exists():
            raise HTTPException(status_code=503, detail="active_model.json says lightgbm but lgbm_model.txt is missing.")
        _model = lgb.Booster(model_file=str(model_path))
    elif _model_type == "logistic_regression":
        model_path = MODEL_DIR / "logreg_model.joblib"
        scaler_path = MODEL_DIR / "logreg_scaler.joblib"
        if not model_path.exists() or not scaler_path.exists():
            raise HTTPException(status_code=503, detail="active_model.json says logistic_regression but logreg files are missing.")
        _model = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)
    else:
        raise HTTPException(status_code=500, detail=f"Unknown active_model_type '{_model_type}' in active_model.json.")

    return _model, _label_encoder


def load_gold() -> pd.DataFrame:
    gold_path = GOLD_DIR / "match_features.parquet"
    if not gold_path.exists():
        raise HTTPException(status_code=503, detail="No gold feature table yet. Run the pipeline first.")
    return pd.read_parquet(gold_path)


def load_teams() -> dict:
    global _teams
    if _teams is None:
        teams_path = GOLD_DIR / "teams.json"
        _teams = json.loads(teams_path.read_text()) if teams_path.exists() else {}
    return _teams


def predict_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Attach home/draw/away probabilities to rows that have complete pre-match features.

    Branches on which model type won training (see load_model()): a LightGBM
    Booster.predict() call returns class probabilities directly, while the
    Logistic Regression path needs the same StandardScaler used at train
    time applied first, then sklearn's predict_proba()."""
    model, le = load_model()
    complete = df.dropna(subset=PRE_MATCH_FEATURES)
    if complete.empty:
        return df.assign(home_win_prob=None, draw_prob=None, away_win_prob=None)

    X = complete[PRE_MATCH_FEATURES]
    if _model_type == "lightgbm":
        probs = model.predict(X)
    else:  # logistic_regression
        X_scaled = _scaler.transform(X)
        probs = model.predict_proba(X_scaled)

    prob_cols = pd.DataFrame(np.asarray(probs), columns=le.classes_, index=complete.index)
    out = df.join(prob_cols)
    return out.rename(columns={"HOME_TEAM": "home_win_prob", "DRAW": "draw_prob", "AWAY_TEAM": "away_win_prob"})


def match_row_to_dict(row) -> dict:
    d = {
        "match_id": int(row["match_id"]),
        "matchday": int(row["matchday"]) if pd.notna(row.get("matchday")) else None,
        "utc_date": str(row["utc_date"]),
        "status": row["status"],
        "home_team_id": int(row["home_team_id"]) if pd.notna(row.get("home_team_id")) else None,
        "home_team": row["home_team_name"],
        "home_team_crest": row.get("home_team_crest"),
        "away_team_id": int(row["away_team_id"]) if pd.notna(row.get("away_team_id")) else None,
        "away_team": row["away_team_name"],
        "away_team_crest": row.get("away_team_crest"),
    }
    if row["status"] == "FINISHED":
        d["home_goals"] = int(row["home_goals"]) if pd.notna(row["home_goals"]) else None
        d["away_goals"] = int(row["away_goals"]) if pd.notna(row["away_goals"]) else None
        d["winner"] = row["winner"]
    else:
        for col, key in [("home_win_prob", "home_win_prob"), ("draw_prob", "draw_prob"), ("away_win_prob", "away_win_prob")]:
            v = row.get(col)
            d[key] = round(float(v), 3) if v is not None and pd.notna(v) else None
    return d


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predictions")
def predictions():
    """Win/draw/loss probabilities for upcoming (SCHEDULED/TIMED) fixtures."""
    df = load_gold()
    upcoming = df[df["status"].isin(["SCHEDULED", "TIMED"])].dropna(subset=PRE_MATCH_FEATURES)
    if upcoming.empty:
        return {"message": "No upcoming fixtures with complete pre-match features yet.", "predictions": []}
    scored = predict_rows(upcoming)
    return {"predictions": [match_row_to_dict(r) for _, r in scored.iterrows()]}


def current_season_only(df: pd.DataFrame) -> pd.DataFrame:
    """Bronze/gold carry 3 merged seasons (for training history); matchweek
    views should only ever show the CURRENT season's fixtures, not have
    e.g. matchday 1 mixing 2024, 2025 and 2026 fixtures together."""
    if "season_start_year" not in df.columns or df["season_start_year"].isna().all():
        return df
    current = df["season_start_year"].max()
    return df[df["season_start_year"] == current]


@app.get("/matchweeks")
def matchweeks():
    """List matchdays (current season only) with fixture counts and date ranges."""
    df = current_season_only(load_gold())
    df = df.dropna(subset=["matchday"])
    grouped = df.groupby("matchday").agg(
        n_fixtures=("match_id", "count"),
        start_date=("utc_date", "min"),
        end_date=("utc_date", "max"),
        n_finished=("status", lambda s: (s == "FINISHED").sum()),
    ).reset_index().sort_values("matchday")
    return {
        "matchweeks": [
            {
                "matchday": int(r["matchday"]),
                "n_fixtures": int(r["n_fixtures"]),
                "n_finished": int(r["n_finished"]),
                "start_date": str(r["start_date"]),
                "end_date": str(r["end_date"]),
            }
            for _, r in grouped.iterrows()
        ]
    }


@app.get("/matchweek/{matchday}")
def matchweek(matchday: int):
    """Fixtures for one matchday (current season only) - predictions for upcoming, results for finished."""
    df = current_season_only(load_gold())
    week = df[df["matchday"] == matchday].sort_values("utc_date")
    if week.empty:
        raise HTTPException(status_code=404, detail=f"No fixtures found for matchday {matchday}")

    upcoming_mask = week["status"].isin(["SCHEDULED", "TIMED"])
    scored_upcoming = predict_rows(week[upcoming_mask]) if upcoming_mask.any() else week[upcoming_mask]
    finished = week[~upcoming_mask]

    combined = pd.concat([scored_upcoming, finished]).sort_values("utc_date")
    return {"matchday": matchday, "fixtures": [match_row_to_dict(r) for _, r in combined.iterrows()]}


@app.get("/team/{team_id}")
def team(team_id: int):
    teams = load_teams()
    t = teams.get(str(team_id))
    if not t:
        raise HTTPException(status_code=404, detail=f"Unknown team id {team_id}")
    return t


@app.get("/h2h/{team_a_id}/{team_b_id}")
def head_to_head(team_a_id: int, team_b_id: int):
    """Historical meetings between two teams, most recent first."""
    df = load_gold()
    mask = (
        ((df["home_team_id"] == team_a_id) & (df["away_team_id"] == team_b_id))
        | ((df["home_team_id"] == team_b_id) & (df["away_team_id"] == team_a_id))
    ) & (df["status"] == "FINISHED")
    meetings = df[mask].sort_values("utc_date", ascending=False)

    history = []
    a_wins = b_wins = draws = 0
    for _, r in meetings.iterrows():
        if r["winner"] == "HOME_TEAM":
            winner_id = int(r["home_team_id"])
        elif r["winner"] == "AWAY_TEAM":
            winner_id = int(r["away_team_id"])
        else:
            winner_id = None
        if winner_id == team_a_id:
            a_wins += 1
        elif winner_id == team_b_id:
            b_wins += 1
        else:
            draws += 1
        history.append({
            "utc_date": str(r["utc_date"]),
            "home_team": r["home_team_name"],
            "away_team": r["away_team_name"],
            "home_goals": int(r["home_goals"]) if pd.notna(r["home_goals"]) else None,
            "away_goals": int(r["away_goals"]) if pd.notna(r["away_goals"]) else None,
            "winner": r["winner"],
        })

    return {
        "team_a_id": team_a_id,
        "team_b_id": team_b_id,
        "meetings_count": len(history),
        "team_a_wins": a_wins,
        "team_b_wins": b_wins,
        "draws": draws,
        "history": history,
    }


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(STATIC_DIR / "index.html"))
