import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# -----------------------
# Load Data
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data/All_stats.csv", encoding="latin1")

    # Ensure consistent season sorting
    if "Season" in df.columns:
        df["Season"] = df["Season"].astype(str)
        df = df.sort_values("Season")
    return df

df = load_data()

# -----------------------
# Identify most recent + previous season
# -----------------------
if "Season" not in df.columns:
    st.error("Missing 'Season' column in All_stats.csv — cannot find previous year data.")
    st.stop()

latest_season = df["Season"].max()
previous_season = sorted(df["Season"].unique())[-2] if len(df["Season"].unique()) > 1 else latest_season

st.markdown(f"### Showing Clutch Stats from **{previous_season}** (for {latest_season} season context)")

# -----------------------
# Prepare data for previous season
# -----------------------
df_prev = df[df["Season"] == previous_season].copy()

# -----------------------
# Default team = highest clutch FGM from last season
# -----------------------
if "CLUTCH_FGM" not in df_prev.columns:
    st.error("No CLUTCH_FGM column found in All_stats.csv.")
    st.stop()

default_team = df_prev.loc[df_prev["CLUTCH_FGM"].idxmax(), "Teams"]
teams_sorted = sorted(df_prev["Teams"].dropna().unique().tolist())

team_name = st.selectbox(
    "Select Team",
    teams_sorted,
    index=teams_sorted.index(default_team) if default_team in teams_sorted else 0,
    key="clutch_team"
)

team_data = df_prev[df_prev["Teams"] == team_name].iloc[0]

# -----------------------
# Name mappings
# -----------------------
stat_name_map = {
    "CLUTCH_FGPERC": "Clutch Field Goal %",
    "CLUTCH_3FGPERC": "Clutch 3 Point %",
    "CLUTCH_FTPERC": "Clutch Free Throw %",
    "CLUTCH_SM": "Clutch Scoring Margin",
    "CLUTCH_REB": "Avg Clutch Rebounds",
    "OPP_CLTCH_REB": "Avg Opponent Clutch Rebounds",
    "CLTCH_OFF_REB": "Avg Clutch Offensive Rebounds",
    "OPP_CLTCH_OFF_REB": "Avg Opp Clutch Off Rebounds",
    "CLTCH_TURN": "Avg Clutch Turnovers",
    "CLTCH_OPP_TURN": "Avg Opp Clutch Turnovers",
    "CLTCH_STL": "Avg Clutch Steals",
    "TOP25_CLUTCH": "Clutch Games vs Top 25",
    "OVERTIME_GAMES": "Overtime Games"
}

stat_pairs = [
    ("CLUTCH_FGPERC", "CLUTCH_FG_RANK"),
    ("CLUTCH_3FGPERC", "CLUTCH_3_RANK"),
    ("CLUTCH_FTPERC", "CLUTCH_FT_RANK"),
    ("CLUTCH_SM", "CLUTCH_SM_RANK"),
    ("CLUTCH_REB", "CLUTCH_REB_RANK"),
    ("OPP_CLTCH_REB", "OPP_CLTCH_REB_RANK"),
    ("CLTCH_OFF_REB", "CLTCH_OFF_REB_RANK"),
    ("OPP_CLTCH_OFF_REB", "OPP_CLTCH_OFF_REB_RANK"),
    ("CLTCH_TURN", "CLTCH_TURN_RANK"),
    ("CLTCH_OPP_TURN", "CLTCH_OPP_TURN_RANK"),
    ("CLTCH_STL", "CLTCH_STL_RANK"),
]

extra_stats = ["TOP25_CLUTCH", "OVERTIME_GAMES"]

# -----------------------
# Clutch Summary Table
# -----------------------
st.subheader(f"{team_name} Clutch Summary ({previous_season})")

summary_rows = []
for stat, rank in stat_pairs:
    summary_rows.append({
        "Stat": stat_name_map.get(stat, stat),
        "Value": team_data.get(stat, np.nan),
        "Rank": team_data.get(rank, np.nan)
    })

for stat in extra_stats:
    summary_rows.append({
        "Stat": stat_name_map.get(stat, stat),
        "Value": team_data.get(stat, np.nan),
        "Rank": None
    })

summary_df = pd.DataFrame(summary_rows)

if summary_df["Value"].isna().all():
    st.warning(f"No clutch data available for {team_name} in {previous_season}.")
else:
    st.dataframe(summary_df, use_container_width=True)

    # -----------------------
    # Visualization: Shooting % Clutch vs Season
    # -----------------------
    st.subheader("Shooting Comparison: Clutch vs Season")

    shooting_stats = ["FG%", "3PT%", "FT%"]
    season_cols = ["FG_PERC", "FG3_PERC", "FT_PERC"]
    clutch_cols = ["CLUTCH_FGPERC", "CLUTCH_3FGPERC", "CLUTCH_FTPERC"]

    season_values = [team_data.get(c, np.nan) * 100 for c in season_cols]
    clutch_values = [team_data.get(c, np.nan) for c in clutch_cols]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=shooting_stats,
        y=season_values,
        name="Season",
        marker_color="lightblue"
    ))
    fig.add_trace(go.Bar(
        x=shooting_stats,
        y=clutch_values,
        name="Clutch",
        marker_color="orange"
    ))
    fig.update_layout(
        barmode="group",
        title=f"{team_name}: Clutch vs Season Shooting ({previous_season})",
        yaxis=dict(title="Percentage"),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)
