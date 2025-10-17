import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# -------------------------------
# 🔧 FILE PATHS
# -------------------------------
SCHEDULE_PATH = "Data/26_March_Madness_Databook/2026 Schedule Simple-Table 1.csv"
DAILY_PATH = "Data/26_March_Madness_Databook/Daily_predictor_data-Table 1.csv"
ALL_STATS_PATH = "Data/26_March_Madness_Databook/All_Stats-THE_TABLE.csv"

# -------------------------------
# 🧠 LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    schedule_df = pd.read_csv(SCHEDULE_PATH)
    daily_df = pd.read_csv(DAILY_PATH)
    all_stats_df = pd.read_csv(ALL_STATS_PATH)
    return schedule_df, daily_df, all_stats_df

schedule_df, daily_df, all_stats_df = load_data()

# -------------------------------
# 🏀 TEAM SELECTION
# -------------------------------
# 👇 Change this to set a new default team
INITIAL_TEAM = "Wisconsin"

team_list = sorted(all_stats_df["Teams"].unique())
default_index = team_list.index(INITIAL_TEAM) if INITIAL_TEAM in team_list else 0

st.title("📅 Schedule Predictor")
selected_team = st.selectbox("Select Team", team_list, index=default_index)

st.write(f"### Selected Team: {selected_team}")

# -------------------------------
# 📊 FILTER FUTURE SCHEDULE
# -------------------------------
team_schedule = schedule_df[schedule_df["Team"] == selected_team].copy()

if team_schedule.empty:
    st.warning(f"No future schedule found for {selected_team}.")
    st.stop()

# -------------------------------
# 🧩 BUILD SIMPLE ML MODEL
# -------------------------------
# We'll train two models:
# 1. LogisticRegression → predict win (binary)
# 2. LinearRegression → predict team points

# Prepare training data
X_cols = ["SM", "Off_eff", "Def_efficiency hybrid", "FG_PERC", "FT_PERC", "3PTM"]  # example features
y_win = (daily_df["Points"] > daily_df["Opp Points"]).astype(int)
X = daily_df[X_cols].fillna(0)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_win, test_size=0.2, random_state=42)
win_model = LogisticRegression(max_iter=500)
win_model.fit(X_train, y_train)

score_model = LinearRegression()
score_model.fit(X_scaled, daily_df["Points"])

# -------------------------------
# 🔮 PREDICTIONS
# -------------------------------
results = []

for _, row in team_schedule.iterrows():
    opponent = row["Opponent"]
    game_date = row["Date"]
    location = row["Location"]

    # Get team + opponent stats
    team_stats = all_stats_df[all_stats_df["Teams"] == selected_team]
    opp_stats = all_stats_df[all_stats_df["Teams"] == opponent]

    if team_stats.empty or opp_stats.empty:
        continue  # Skip if missing stats

    # Create combined input (team minus opponent)
    diff = (team_stats[X_cols].values - opp_stats[X_cols].values)
    diff_scaled = scaler.transform(diff)

    # Predict win probability and score
    win_prob = win_model.predict_proba(diff_scaled)[0][1]
    team_score = score_model.predict(diff_scaled)[0]
    opp_score = team_score - np.random.uniform(3, 8) if win_prob > 0.5 else team_score + np.random.uniform(3, 8)

    results.append({
        "Date": game_date,
        "Opponent": opponent,
        "Location": location,
        "Win_Confidence": f"{win_prob*100:.1f}%",
        "Predicted_Score": f"{selected_team} {team_score:.0f} - {opponent} {opp_score:.0f}"
    })

# -------------------------------
# 🏁 DISPLAY RESULTS
# -------------------------------
if results:
    results_df = pd.DataFrame(results)
    st.subheader(f"Predicted Upcoming Games for {selected_team}")
    st.dataframe(results_df, use_container_width=True)
else:
    st.info("No predictions available (missing data for some opponents).")

st.markdown("---")
st.caption("📊 This model uses basic statistical differentials. Future iterations can incorporate advanced ML and real betting market features.")
