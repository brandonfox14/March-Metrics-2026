import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------
# Load Data
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data/26_March_Madness_Databook/All_Stats-THE_TABLE.csv", encoding="latin1")
    # Replace NaNs with 0s to start the season
    df = df.fillna(0)
    return df

df = load_data()

# -----------------------
# Team Selection
# -----------------------
teams_sorted = sorted(df["Teams"].dropna().unique().tolist())
if not teams_sorted:
    st.error("No teams found in the dataset.")
    st.stop()

default_team = teams_sorted[0]
team_name = st.selectbox("Select Team", teams_sorted, index=teams_sorted.index(default_team))

team_data = df[df["Teams"] == team_name]

if team_data.empty:
    st.warning(f"No data available for {team_name}.")
    st.stop()

team_data = team_data.iloc[0]  # use the single team’s row

# -----------------------
# Stat Categories
# -----------------------
scoring_stats = [
    ("PTS", "Points Per Game"),
    ("OPP_PTS", "Opponent Points Per Game"),
    ("SCORING_MARGIN", "Scoring Margin"),
    ("PPG_RANK", "Points Rank"),
    ("OPP_PPG_RANK", "Opponent Points Rank")
]

offense_stats = [
    ("FG_PERC", "Field Goal %"),
    ("FG3_PERC", "3PT %"),
    ("FT_PERC", "Free Throw %"),
    ("AST", "Assists per Game"),
    ("TO", "Turnovers per Game"),
    ("OREB", "Offensive Rebounds per Game")
]

defense_stats = [
    ("DREB", "Defensive Rebounds per Game"),
    ("STL", "Steals per Game"),
    ("BLK", "Blocks per Game"),
    ("DEF_RTG", "Defensive Rating"),
    ("OPP_FG_PERC", "Opponent FG%"),
    ("OPP_3P_PERC", "Opponent 3PT%")
]

extra_stats = [
    ("CONF", "Conference"),
    ("LEAGUE_RANK", "Overall League Rank"),
    ("COACH_VALUE", "Coach Value"),
    ("SOS_RANK", "Strength of Schedule Rank")
]

# -----------------------
# Helper function
# -----------------------
def build_summary_table(stat_list, title):
    summary_rows = []
    for stat, label in stat_list:
        value = team_data.get(stat, np.nan)
        if isinstance(value, (int, float)) and np.isnan(value):
            value = 0
        summary_rows.append({"Stat": label, "Value": value})
    df_out = pd.DataFrame(summary_rows)
    st.subheader(title)
    st.dataframe(df_out, use_container_width=True)
    return df_out

# -----------------------
# Display Team Breakdown
# -----------------------
st.title(f"{team_name} – Team Breakdown")

st.markdown("### Scoring Overview")
build_summary_table(scoring_stats, "Scoring Statistics")

st.markdown("### Offensive Overview")
build_summary_table(offense_stats, "Offensive Statistics")

st.markdown("### Defensive Overview")
build_summary_table(defense_stats, "Defensive Statistics")

st.markdown("### Extras / Context")
build_summary_table(extra_stats, "Additional Info")

# -----------------------
# Visualization – Scoring Comparison vs Conference & League
# -----------------------
try:
    st.subheader("Scoring Comparison vs Conference & League")

    conf_name = team_data.get("CONF", None)
    if conf_name:
        conf_avg = df[df["CONF"] == conf_name].mean(numeric_only=True)
    else:
        conf_avg = pd.Series(0, index=df.select_dtypes(include=np.number).columns)

    league_avg = df.mean(numeric_only=True)

    metrics = ["PTS", "OPP_PTS", "SCORING_MARGIN"]
    team_values = [team_data.get(m, 0) for m in metrics]
    conf_values = [conf_avg.get(m, 0) for m in metrics]
    league_values = [league_avg.get(m, 0) for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=metrics, y=team_values, name=team_name, marker_color="red"))
    fig.add_trace(go.Bar(x=metrics, y=conf_values, name="Conference Avg", marker_color="blue"))
    fig.add_trace(go.Bar(x=metrics, y=league_values, name="League Avg", marker_color="gray"))

    fig.update_layout(
        barmode="group",
        title=f"{team_name} Scoring vs Conference and League Averages",
        yaxis=dict(title="Per Game"),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"Could not render visualization: {e}")
