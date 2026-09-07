"""
Phase 4 - Dashboard: this week's (and beyond) predictions.

A small Streamlit app over the live /predictions endpoint. Reads the
API base URL from PL_PREDICTOR_API_URL (defaults to the deployed
Render service) so it can point at localhost during development.
"""

import os
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("PL_PREDICTOR_API_URL", "https://plprediction-met6.onrender.com")

st.set_page_config(page_title="PL Match Predictor", page_icon="⚽", layout="wide")

st.title("⚽ Premier League Match Predictor")
st.caption(
    "Leakage-checked LightGBM model, trained on a PySpark/Delta Lake medallion "
    "pipeline (Databricks) or the local pandas pipeline, served live."
)


@st.cache_data(ttl=300)
def load_predictions(api_url: str):
    resp = requests.get(f"{api_url}/predictions", timeout=30)
    resp.raise_for_status()
    return resp.json()


with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=API_URL)
    st.caption(
        "The free-tier API sleeps when idle - the first load after "
        "inactivity can take 30-60s to wake it up."
    )
    if st.button("Refresh predictions"):
        st.cache_data.clear()

try:
    data = load_predictions(api_url)
except Exception as e:
    st.error(f"Couldn't reach the prediction API at {api_url}: {e}")
    st.stop()

predictions = data.get("predictions", [])
if not predictions:
    st.warning(data.get("message", "No upcoming predictions available right now."))
    st.stop()

df = pd.DataFrame(predictions)
df["utc_date"] = pd.to_datetime(df["utc_date"])
df["fixture"] = df["home_team"] + " vs " + df["away_team"]
df["predicted_outcome"] = df[["home_win_prob", "draw_prob", "away_win_prob"]].idxmax(axis=1).map({
    "home_win_prob": "Home win",
    "draw_prob": "Draw",
    "away_win_prob": "Away win",
})
df["confidence"] = df[["home_win_prob", "draw_prob", "away_win_prob"]].max(axis=1)

now = pd.Timestamp.now(tz=timezone.utc)
upcoming = df[df["utc_date"] >= now - pd.Timedelta(days=1)].sort_values("utc_date")

col1, col2, col3 = st.columns(3)
col1.metric("Upcoming fixtures", len(upcoming))
col2.metric("Predicted home wins", int((upcoming["predicted_outcome"] == "Home win").sum()))
col3.metric("Predicted away wins", int((upcoming["predicted_outcome"] == "Away win").sum()))

st.subheader("This week")
gameweek_cutoff = now + pd.Timedelta(days=7)
this_week = upcoming[upcoming["utc_date"] <= gameweek_cutoff]

if this_week.empty:
    st.info("No fixtures in the next 7 days - showing the next batch instead.")
    this_week = upcoming.head(10)

for _, row in this_week.iterrows():
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    c1.markdown(f"**{row['fixture']}**  \n{row['utc_date'].strftime('%a %b %d, %H:%M UTC')}")
    c2.metric("Home", f"{row['home_win_prob']:.0%}")
    c3.metric("Draw", f"{row['draw_prob']:.0%}")
    c4.metric("Away", f"{row['away_win_prob']:.0%}")

st.subheader("All upcoming fixtures")
st.dataframe(
    upcoming[["utc_date", "fixture", "home_win_prob", "draw_prob", "away_win_prob", "predicted_outcome", "confidence"]]
    .rename(columns={
        "utc_date": "Date", "fixture": "Fixture",
        "home_win_prob": "Home win", "draw_prob": "Draw", "away_win_prob": "Away win",
        "predicted_outcome": "Predicted", "confidence": "Confidence",
    }),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Most confident predictions")
st.bar_chart(
    upcoming.nlargest(10, "confidence").set_index("fixture")["confidence"]
)

st.caption(
    "Model accuracy and log loss are validated against naive baselines on a "
    "held-out, time-ordered test set - see the project README for current numbers. "
    "Early-season predictions especially should be read as provisional."
)
