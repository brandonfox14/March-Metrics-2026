import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="Schedule Predictor", layout="wide")
st.title("📅 Schedule Predictor")

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
@st.cache_data
def load_data():
    all_stats = pd.read_csv("Data/26_March_Madness_Databook/All_Stats-THE_TABLE.csv", encoding="latin1")
    daily_df = pd.read_csv("Data/26_March_Madness_Databook/Daily_predictor_data-Table 1.csv", encoding="latin1")
    schedule_df = pd.read_csv("Data/26_March_Madness_Databook/2026 Schedule Simple-Table 1.csv", encoding="latin1")
    return all_stats, daily_df, schedule_df

all_stats, daily_df, schedule_df = load_data()

# -------------------------------------------------------------
# CLEAN & PREP DATA
# -------------------------------------------------------------
# Convert all date formats to datetime and sort properly
for col in ["Date", "date", "Game_Date"]:
    if col in schedule_df.columns:
        schedule_df["Date"] = pd.to_datetime(schedule_df[col], errors="coerce")
        break
else:
    schedule_df["Date"] = pd.to_datetime(schedule_df.iloc[:, 0], errors="coerce")

schedule_df = schedule_df.sort_values("Date").reset_index(drop=True)

# -------------------------------------------------------------
# TEAM SELECTION (Initial = Wisconsin)
# -------------------------------------------------------------
DEFAULT_TEAM = "Wisconsin"  # 👈 Change this line to adjust the initial team

team_list = sorted(all_stats["Team"].dropna().unique())
selected_team = st.selectbox("Select a Team", options=team_list, index=team_list.index(DEFAULT_TEAM))

# -------------------------------------------------------------
# PREP MACHINE LEARNING
# -------------------------------------------------------------
# Identify target and features
numeric_cols = daily_df.select_dtypes(include=[np.number]).columns.tolist()

# Handle missing values
daily_df = daily_df.fillna(0)

# Predict whether team wins (binary)
if "Points" in daily_df.columns and "Opp Points" in daily_df.columns:
    daily_df["Win"] = (daily_df["Points"] > daily_df["Opp Points"]).astype(int)
else:
    st.error("Daily predictor data missing 'Points' or 'Opp Points' columns.")
    st.stop()

# Feature matrix
X = daily_df[numeric_cols]
y = daily_df["Win"]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest Classifier for Win Probability
win_model = RandomForestClassifier(n_estimators=300, random_state=42)
win_model.fit(X_train, y_train)

# Random Forest Regressors for projected scores
score_model_for = RandomForestRegressor(n_estimators=300, random_state=42)
score_model_against = RandomForestRegressor(n_estimators=300, random_state=42)
score_model_for.fit(X_scaled, daily_df["Points"])
score_model_against.fit(X_scaled, daily_df["Opp Points"])

# -------------------------------------------------------------
# GENERATE PREDICTIONS FOR UPCOMING GAMES
# -------------------------------------------------------------
# Filter schedule for selected team
mask = (schedule_df["Team"] == selected_team) | (schedule_df["Opponent"] == selected_team)
team_schedule = schedule_df.loc[mask].copy().sort_values("Date")

if team_schedule.empty:
    st.warning(f"No scheduled games found for {selected_team}.")
    st.stop()

# Pull last known averages for this team
if selected_team in all_stats["Team"].values:
    team_stats = all_stats[all_stats["Team"] == selected_team].select_dtypes(include=[np.number]).mean().fillna(0)
else:
    team_stats = pd.Series(np.zeros(len(numeric_cols)), index=numeric_cols)

# Build prediction DataFrame
predictions = []
for _, row in team_schedule.head(3).iterrows():
    opponent = row["Opponent"] if row["Team"] == selected_team else row["Team"]
    game_date = row["Date"]
    home_away = "Home" if row["Team"] == selected_team else "Away"

    # Build input vector (extend or trim to match model input)
    x_input = pd.DataFrame([team_stats], columns=numeric_cols).fillna(0)
    x_scaled = scaler.transform(x_input)

    win_prob = win_model.predict_proba(x_scaled)[0][1]
    proj_for = score_model_for.predict(x_scaled)[0]
    proj_against = score_model_against.predict(x_scaled)[0]

    predictions.append({
        "Date": game_date.strftime("%Y-%m-%d"),
        "Opponent": opponent,
        "Location": home_away,
        "Win Confidence": f"{win_prob*100:.1f}%",
        "Projected Score": f"{selected_team} {proj_for:.0f} - {opponent} {proj_against:.0f}"
    })

pred_df = pd.DataFrame(predictions)

# -------------------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------------------
st.markdown(f"### 🏀 Upcoming Games for {selected_team}")
st.dataframe(pred_df, use_container_width=True)

st.markdown(
    """
    **Notes:**
    - *Win Confidence* = Probability of winning based on full historical metrics  
    - *Projected Score* = Expected final score using Random Forest regressors  
    - *Change the default team* by editing `DEFAULT_TEAM` near the top of this file.
    """
)
