# APP/Pages/6_Todays_Games.py
"""
Today's Games page.

- Retrains regression models each run from Daily_predictor_data-Table 1.csv.
- Uses numeric columns from schedule transfer as features.
- Predicts Points and Opp Points for each matchup (no confidence intervals).
- Sorts and displays upcoming games with a per-game expander showing team details and top-50 ranks.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime

# -------------------------
# CONFIG
# -------------------------
BASE = "Data/26_March_Madness_Databook"
SCHEDULE_FILE = os.path.join(BASE, "2026 Schedule Transfer-Table 1.csv")
DAILY_FILE = os.path.join(BASE, "Daily_predictor_data-Table 1.csv")

st.set_page_config(page_title="Today's Games", layout="wide")
st.title("Today's Games — Predictions")

# -------------------------
# LOAD CSVS
# -------------------------
@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="latin1")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error reading {path}: {e}")
        return None

schedule_df = load_csv(SCHEDULE_FILE)
daily_df = load_csv(DAILY_FILE)

if schedule_df is None:
    st.error(f"Schedule transfer file not found at: {SCHEDULE_FILE}")
    st.stop()
if daily_df is None:
    st.error(f"Daily predictor file not found at: {DAILY_FILE}")
    st.stop()

st.info("CSV files loaded.")

# -------------------------
# Helper functions
# -------------------------
def interpret_han(val):
    if pd.isna(val):
        return None
    s = str(val).strip().upper()
    if s in ["H", "HOME"]:
        return "Home"
    if s in ["A", "AWAY"]:
        return "Away"
    if "NEUTRAL" in s or s == "N":
        return "Neutral"
    return None

def parse_date_series(s):
    parsed = pd.to_datetime(s.astype(str).str.strip(), errors="coerce", infer_datetime_format=True)
    if parsed.isna().mean() > 0.25:
        parsed = pd.to_datetime(s.astype(str).str.strip(), errors="coerce", dayfirst=True, infer_datetime_format=True)
    return parsed

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------------------
# Parse dates and clean schedule
# -------------------------
date_col = find_col(schedule_df, ["Date", "Game_Date", "Game Date"])
if date_col is None:
    date_col = schedule_df.columns[0]

schedule_df["__Date_parsed"] = parse_date_series(schedule_df[date_col])
schedule_df = schedule_df.dropna(subset=["__Date_parsed"]).reset_index(drop=True)

# Identify key columns
team_col = find_col(schedule_df, ["Team", "Teams"])
opp_col = find_col(schedule_df, ["Opponent", "Opp"])
han_col = find_col(schedule_df, ["HAN", "Home/Away", "Location"])
top25_col = find_col(schedule_df, ["Top 25 Opponent", "Top25"])
mm_col = find_col(schedule_df, ["March Madness Opponent", "March Madness"])
conf_col = find_col(schedule_df, ["Conference"])
opp_conf_col = find_col(schedule_df, ["Opponent Conference"])
coach_col = find_col(schedule_df, ["Coach Name", "Coach"])
opp_coach_col = find_col(schedule_df, ["Opponent Coach"])

# Ensure team/opp exist
if team_col is None or opp_col is None:
    st.error("Schedule CSV missing Team or Opponent columns.")
    st.stop()

# -------------------------
# Prepare training data
# -------------------------
points_col = find_col(daily_df, ["Points"])
opp_points_col = find_col(daily_df, ["Opp Points", "OppPoints", "Opp_Points"])

if points_col is None or opp_points_col is None:
    st.error("Daily predictor file must have 'Points' and 'Opp Points'.")
    st.stop()

# Use numeric features from daily_df
numeric_cols = daily_df.select_dtypes(include=[np.number]).columns.tolist()
train_features = [c for c in numeric_cols if c not in [points_col, opp_points_col]]
if len(train_features) == 0:
    st.error("No numeric training features detected.")
    st.stop()

X_train = daily_df[train_features].fillna(0).values
y_points = daily_df[points_col].fillna(0).values
y_opp = daily_df[opp_points_col].fillna(0).values

# Train models
reg_points = RandomForestRegressor(n_estimators=200, random_state=42)
reg_opp = RandomForestRegressor(n_estimators=200, random_state=42)
reg_points.fit(X_train, y_points)
reg_opp.fit(X_train, y_opp)

st.write("Models trained.")

# -------------------------
# Build predictions
# -------------------------
predictions = []

def conf_priority(conf):
    if pd.isna(conf):
        return 10
    c = str(conf).strip().upper()
    if c == "SEC":
        return 2
    if c in ["BIG TEN", "B1G", "BIGTEN"]:
        return 3
    if c in ["BIG 12", "BIG12"]:
        return 4
    if c == "ACC":
        return 5
    return 9

# Detect rank columns (1-50)
rank_cols = [c for c in schedule_df.columns if "_RANK" in c.upper() or c.upper().endswith(" RANK") or c.upper().endswith("_Rank")]

for idx, row in schedule_df.iterrows():
    t = str(row[team_col]).strip()
    o = str(row[opp_col]).strip()
    
    # Build feature vector for model (only numeric columns present in daily_df)
    feat_vec = []
    for f in train_features:
        if f in row:
            val = row[f]
            try:
                val = float(val)
            except:
                val = 0.0
        else:
            val = 0.0
        feat_vec.append(val)
    feat_vec = np.array(feat_vec).reshape(1, -1)
    
    # Predict
    pred_points = float(reg_points.predict(feat_vec)[0])
    pred_opp_points = float(reg_opp.predict(feat_vec)[0])
    
    # Metadata for dropdown
    def team_meta(team_prefix):
        meta = {}
        meta["Coach"] = row[coach_col] if team_prefix == t else row[opp_coach_col] if opp_coach_col in row else ""
        meta["Wins"] = row.get("Wins", "")
        meta["Losses"] = row.get("Losses", "")
        meta["Conference"] = row.get(conf_col if team_prefix==t else opp_conf_col, "")
        # Top50
        top50 = {}
        for rc in rank_cols:
            val = row.get(rc)
            try:
                val_int = int(val)
            except:
                val_int = 51
            if val_int >=1 and val_int <=50:
                top50[rc] = val_int
        meta["Top50"] = top50
        return meta
    
    predictions.append({
        "Date": row["__Date_parsed"],
        "Team": t,
        "Opponent": o,
        "Pred_Points": round(pred_points),
        "Pred_Opp_Points": round(pred_opp_points),
        "Team_Meta": team_meta(t),
        "Opp_Meta": team_meta(o),
        "Top25": bool(row[top25_col]) if top25_col in row else False,
        "MarchMadness": bool(row[mm_col]) if mm_col in row else False,
        "ConfPriority": min(conf_priority(row[conf_col]), conf_priority(row[opp_conf_col]))
    })

# -------------------------
# Sort by Top25 > MarchMadness > Conf > Date
# -------------------------
pred_df = pd.DataFrame(predictions)
pred_df = pred_df.sort_values(
    by=["Top25", "MarchMadness", "ConfPriority", "Date"],
    ascending=[False, False, True, True]
).reset_index(drop=True)

# -------------------------
# Display games
# -------------------------
st.subheader("Predicted Upcoming Games")

for idx, row in pred_df.iterrows():
    date_str = row["Date"].strftime("%b %d, %Y") if pd.notna(row["Date"]) else "TBD"
    game_label = f"{row['Team']} vs {row['Opponent']} — {date_str} — Pred: {row['Pred_Points']} - {row['Pred_Opp_Points']}"
    with st.expander(game_label, expanded=False):
        # Display team metadata
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**{row['Team']}**")
            st.write(f"Coach: {row['Team_Meta']['Coach']}")
            st.write(f"Wins: {row['Team_Meta']['Wins']}")
            st.write(f"Losses: {row['Team_Meta']['Losses']}")
            st.write(f"Conference: {row['Team_Meta']['Conference']}")
        with col2:
            st.write(f"**{row['Opponent']}**")
            st.write(f"Coach: {row['Opp_Meta']['Coach']}")
            st.write(f"Wins: {row['Opp_Meta']['Wins']}")
            st.write(f"Losses: {row['Opp_Meta']['Losses']}")
            st.write(f"Conference: {row['Opp_Meta']['Conference']}")

        st.markdown("---")
        # Top50 table
        all_top50_keys = sorted(set(list(row['Team_Meta']['Top50'].keys()) + list(row['Opp_Meta']['Top50'].keys())))
        if all_top50_keys:
            top50_data = []
            for k in all_top50_keys:
                top50_data.append({
                    "Category": k,
                    row['Team']: row['Team_Meta']['Top50'].get(k, ""),
                    row['Opponent']: row['Opp_Meta']['Top50'].get(k, "")
                })
            st.markdown("#### Top-50 in the Nation")
            st.dataframe(pd.DataFrame(top50_data), use_container_width=True)
        else:
            st.write("No top-50 ranks available for these teams.")

# -------------------------
# CSV download
# -------------------------
out_csv = pred_df[["Date","Team","Opponent","Pred_Points","Pred_Opp_Points"]].copy()
out_csv["Date"] = out_csv["Date"].dt.strftime("%m-%d-%Y")
st.download_button("Download predictions CSV", out_csv.to_csv(index=False), "todays_games_predictions.csv", "text/csv")
