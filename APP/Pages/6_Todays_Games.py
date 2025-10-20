# APP/Pages/6_Todays_Games.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Data loading
# -------------------------------
@st.cache_data
def load_data():
    schedule = pd.read_csv("Data/26_March_Madness_Databook/All_Stats-THE_TABLE.csv")
    results = pd.read_csv("Data/26_March_Madness_Databook/Daily_predictor_data-Table 1.csv")

    # Replace early-season NAs with 51+
    schedule = schedule.fillna(51)
    results = results.fillna(51)

    # Standardize column names
    schedule.columns = schedule.columns.str.strip()
    results.columns = results.columns.str.strip()

    # Convert date columns
    for df in [schedule, results]:
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Ensure numeric types where possible
    for df in [schedule, results]:
        for c in df.columns:
            if df[c].dtype == "object":
                try:
                    df[c] = pd.to_numeric(df[c])
                except Exception:
                    pass

    return schedule, results


schedule_df, results_df = load_data()

# -------------------------------
# Page layout
# -------------------------------
st.title("Today's Projected Games")

# Verify required columns
required_cols = {"Team", "Opponent", "Score", "Opponent_Score"}
missing = required_cols - set(results_df.columns)
if missing:
    st.error(f"Missing expected columns in results data: {missing}")
    st.stop()

# -------------------------------
# Model training
# -------------------------------
# Prepare features and targets
exclude_cols = ["Date", "Team", "Opponent", "Score", "Opponent_Score"]
feature_cols = [c for c in results_df.columns if c not in exclude_cols]

X = results_df[feature_cols]
y_team = results_df["Score"]
y_opp = results_df["Opponent_Score"]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train two separate regressors for projected points
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_team, test_size=0.2, random_state=42)
model_team = RandomForestRegressor(n_estimators=300, random_state=42)
model_team.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_opp, test_size=0.2, random_state=42)
model_opp = RandomForestRegressor(n_estimators=300, random_state=42)
model_opp.fit(X_train, y_train)

# -------------------------------
# Predict for today's schedule
# -------------------------------
if "Date" not in schedule_df.columns or "Team" not in schedule_df.columns or "Opponent" not in schedule_df.columns:
    st.error("Schedule file must include columns: Date, Team, Opponent.")
    st.stop()

# Sort dates chronologically
schedule_df["Date"] = pd.to_datetime(schedule_df["Date"], errors="coerce")
schedule_df = schedule_df.sort_values(by="Date")

today = pd.Timestamp.now().normalize()
todays_games = schedule_df[schedule_df["Date"] >= today]

if todays_games.empty:
    st.info("No upcoming games found in schedule data.")
    st.stop()

feature_cols_sched = [c for c in todays_games.columns if c not in ["Date", "Team", "Opponent"]]
X_today = todays_games[feature_cols_sched]
X_today_scaled = scaler.transform(X_today)

pred_team = model_team.predict(X_today_scaled)
pred_opp = model_opp.predict(X_today_scaled)

todays_games["Predicted_Team_Score"] = pred_team.round(1)
todays_games["Predicted_Opp_Score"] = pred_opp.round(1)

# -------------------------------
# Display projected games
# -------------------------------
for _, row in todays_games.iterrows():
    date_str = row["Date"].strftime("%Y-%m-%d") if pd.notna(row["Date"]) else "TBD"
    team = row["Team"]
    opp = row["Opponent"]
    st.markdown(f"**{team} vs {opp}** — {date_str}")
    st.write(f"Projected Score: {team} {row['Predicted_Team_Score']} - {opp} {row['Predicted_Opp_Score']}")
    st.divider()
