import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import os

st.title("Today's Games Prediction Dashboard")

# -----------------------
# Load Data
# -----------------------
@st.cache_data
def load_data():
    possible_paths = [
        "Data/26_March_Madness_Databook/Schedule_Transfer.csv",
        "./Data/26_March_Madness_Databook/Schedule_Transfer.csv",
        "APP/Data/26_March_Madness_Databook/Schedule_Transfer.csv",
    ]
    schedule_path = next((p for p in possible_paths if os.path.exists(p)), None)

    possible_daily_paths = [
        "Data/26_March_Madness_Databook/Daily_predictor_data-Table 1.csv",
        "./Data/26_March_Madness_Databook/Daily_predictor_data-Table 1.csv",
        "APP/Data/26_March_Madness_Databook/Daily_predictor_data-Table 1.csv",
    ]
    daily_path = next((p for p in possible_daily_paths if os.path.exists(p)), None)

    if not schedule_path:
        st.error("❌ Could not find Schedule_Transfer.csv — please check your file path.")
        return None, None
    if not daily_path:
        st.error("❌ Could not find Daily_predictor_data-Table 1.csv — please check your file path.")
        return None, None

    schedule_df = pd.read_csv(schedule_path, encoding="latin1").fillna(51)
    daily_df = pd.read_csv(daily_path, encoding="latin1").fillna(51)

    # Ensure expected columns exist
    for col in ["Team", "Opponent", "Points", "Opp Points"]:
        if col not in schedule_df.columns:
            schedule_df[col] = np.nan
        if col not in daily_df.columns:
            daily_df[col] = np.nan

    return schedule_df, daily_df


schedule_df, daily_df = load_data()
if schedule_df is None or daily_df is None:
    st.stop()

# -----------------------
# Build and retrain model
# -----------------------
@st.cache_resource
def train_model(daily_df):
    train_df = daily_df.dropna(subset=["Points", "Opp Points"]).copy()
    exclude_cols = ["Team", "Opponent", "Date", "Points", "Opp Points"]
    numeric_cols = [c for c in train_df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(train_df[c])]

    if len(numeric_cols) == 0:
        st.error("No numeric features found in daily predictor data.")
        st.stop()

    X = train_df[numeric_cols]
    y = train_df["Points"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, scaler, numeric_cols


model, scaler, feature_cols = train_model(daily_df)

# -----------------------
# Predict Today's Games
# -----------------------
schedule_df = schedule_df.copy()
schedule_df["Date"] = pd.to_datetime(schedule_df.get("Date", pd.NaT), errors="coerce")

predict_df = schedule_df.copy()
X_pred = predict_df[feature_cols].copy().fillna(51)

try:
    X_pred_scaled = scaler.transform(X_pred)
    predict_df["Predicted Points"] = model.predict(X_pred_scaled).round(1)
except Exception as e:
    st.error(f"Error during prediction: {e}")
    st.stop()

# -----------------------
# Display Predicted Results
# -----------------------
st.subheader("Predicted Results for Today's Games")
st.dataframe(
    predict_df[["Team", "Opponent", "Predicted Points", "Points", "Opp Points"]]
        .sort_values("Predicted Points", ascending=False),
    use_container_width=True
)

# -----------------------
# Visualization
# -----------------------
st.subheader("Predicted Points Distribution")

fig = go.Figure()
fig.add_trace(go.Bar(
    x=predict_df["Team"],
    y=predict_df["Predicted Points"],
    name="Predicted Points",
    marker_color="steelblue"
))
fig.update_layout(
    template="plotly_white",
    title="Predicted Points for Today's Games",
    xaxis_title="Team",
    yaxis_title="Predicted Points",
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Matchup Breakdown
# -----------------------
st.subheader("Matchup Breakdown")

for _, row in predict_df.iterrows():
    team = row["Team"]
    opp = row["Opponent"]
    date = row.get("Date", None)

    if pd.isna(team) or pd.isna(opp):
        continue

    team_lower, opp_lower = str(team).lower(), str(opp).lower()
    date_component = pd.to_datetime(date).date() if not pd.isna(date) else None
    matchup_key = (team_lower, opp_lower, date_component)

    date_str = f"{date_component.strftime('%b %d, %Y')}" if date_component else "Date TBD"
    st.markdown(f"**{team} vs {opp}** — {date_str}")
    st.write(f"Predicted Points: {row['Predicted Points']}")
    if not pd.isna(row.get('Points')):
        st.write(f"Actual Points: {row['Points']}")
    st.divider()
