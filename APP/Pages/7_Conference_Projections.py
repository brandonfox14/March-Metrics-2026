import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# ------------------------------------------
# CONFIG
# ------------------------------------------
BASE = "Data/26_March_Madness_Databook"
SCHEDULE_FILE = os.path.join(BASE, "2026 Schedule Simple-Table 1.csv")
DAILY_FILE = os.path.join(BASE, "Daily_predictor_data-Table 1.csv")
ALL_STATS_FILE = os.path.join(BASE, "All_Stats-THE_TABLE.csv")

st.set_page_config(page_title="Conference Predictor", layout="wide")
st.title("🏀 Conference Predictor")
st.write(
    "This page projects standings and win totals for each conference. "
    "It uses the same predictive model from the Schedule Predictor to estimate each team's expected record."
)

# ------------------------------------------
# SAFE DATA LOADERS
# ------------------------------------------
@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        return None
    df = pd.read_csv(path, encoding="latin1")
    df.columns = df.columns.str.strip()
    return df

schedule_df = load_csv(SCHEDULE_FILE)
daily_df = load_csv(DAILY_FILE)
all_stats = load_csv(ALL_STATS_FILE)

if schedule_df is None or daily_df is None or all_stats is None:
    st.stop()

# ------------------------------------------
# BASIC SETUP
# ------------------------------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

team_col = find_col(all_stats, ["Teams", "Team", "team"])
conf_col = find_col(all_stats, ["Conference", "Conf", "conference"])
coach_col = find_col(all_stats, ["Coach", "Coach Name"])
wins_col = find_col(all_stats, ["Wins", "W", "Win"])
losses_col = find_col(all_stats, ["Losses", "L", "Loss"])

if team_col is None or conf_col is None:
    st.error("Missing team or conference columns in All_Stats.")
    st.stop()

# ------------------------------------------
# BUILD TRAINING DATA (same ML structure)
# ------------------------------------------
numeric_cols = all_stats.select_dtypes(include=[np.number]).columns.tolist()

team_numeric_lookup = {}
for _, r in all_stats.iterrows():
    t = str(r[team_col]).strip()
    team_numeric_lookup[t] = pd.to_numeric(r[numeric_cols], errors="coerce").fillna(0.0).values

# Daily predictor training
daily_team_col = find_col(daily_df, ["Team", "Teams", "team"])
daily_opp_col = find_col(daily_df, ["Opponent", "Opp", "opponent"])
daily_points_col = find_col(daily_df, ["Points"])
daily_opp_points_col = find_col(daily_df, ["Opp Points", "Opp_Points"])

train_rows = []
for _, r in daily_df.iterrows():
    tname = str(r[daily_team_col]).strip() if pd.notna(r[daily_team_col]) else None
    oname = str(r[daily_opp_col]).strip() if pd.notna(r[daily_opp_col]) else None
    if not tname or not oname:
        continue
    if tname not in team_numeric_lookup or oname not in team_numeric_lookup:
        continue

    tnum = team_numeric_lookup[tname]
    onum = team_numeric_lookup[oname]
    feat = np.concatenate([tnum, onum, tnum - onum])

    try:
        pts = float(r[daily_points_col])
        opp = float(r[daily_opp_points_col])
    except Exception:
        continue
    win = 1 if pts > opp else 0

    train_rows.append({"feat": feat, "pts": pts, "opp": opp, "win": win})

if len(train_rows) == 0:
    st.error("No valid daily predictor rows found for training.")
    st.stop()

X = np.vstack([r["feat"] for r in train_rows])
Y_points = np.column_stack([[r["pts"] for r in train_rows], [r["opp"] for r in train_rows]])
Y_win = np.array([r["win"] for r in train_rows])

# Fit models
rf_multi = MultiOutputRegressor(RandomForestRegressor(n_estimators=500, random_state=42))
rf_multi.fit(X, Y_points)

clf = RandomForestClassifier(n_estimators=400, random_state=42)
clf.fit(X, Y_win)

# ------------------------------------------
# BUILD CONFERENCE TABS
# ------------------------------------------
unique_confs = sorted(all_stats[conf_col].dropna().unique().tolist())
tabs = st.tabs(unique_confs)

# ------------------------------------------
# SIMULATION LOOP PER CONFERENCE
# ------------------------------------------
schedule_team_col = find_col(schedule_df, ["Team", "Teams"])
schedule_opp_col = find_col(schedule_df, ["Opponent", "Opp"])
wins_col_sched = find_col(schedule_df, ["Wins", "W"])
losses_col_sched = find_col(schedule_df, ["Losses", "L"])

# Average total points for normalization
avg_total = (daily_df[daily_points_col].mean() + daily_df[daily_opp_points_col].mean()) / 2

for conf_name, tab in zip(unique_confs, tabs):
    with tab:
        st.subheader(f"{conf_name} Outlook")

        conf_teams = all_stats[all_stats[conf_col] == conf_name][team_col].dropna().unique().tolist()
        results = []

        for team in conf_teams:
            if team not in team_numeric_lookup:
                continue

            team_wins = float(all_stats.loc[all_stats[team_col] == team, wins_col].fillna(0).iloc[0]) if wins_col else 0
            team_losses = float(all_stats.loc[all_stats[team_col] == team, losses_col].fillna(0).iloc[0]) if losses_col else 0

            team_sched = schedule_df[schedule_df[schedule_team_col] == team]
            expected_wins = 0

            for _, g in team_sched.iterrows():
                opp = str(g[schedule_opp_col]).strip()
                if opp not in team_numeric_lookup:
                    continue

                feat = np.concatenate([
                    team_numeric_lookup[team],
                    team_numeric_lookup[opp],
                    team_numeric_lookup[team] - team_numeric_lookup[opp]
                ])
                feat = feat.reshape(1, -1)
                win_prob = clf.predict_proba(feat)[0][1]
                expected_wins += win_prob

            proj_total_wins = team_wins + expected_wins
            proj_losses = max(0, team_losses + (len(team_sched) - expected_wins))
            results.append({
                "Team": team,
                "Conference": conf_name,
                "Current Wins": int(team_wins),
                "Current Losses": int(team_losses),
                "Projected Wins": round(proj_total_wins, 1),
                "Projected Losses": round(proj_losses, 1),
                "Δ Wins": round(proj_total_wins - team_wins, 1)
            })

        if not results:
            st.write("No valid teams in this conference.")
            continue

        df = pd.DataFrame(results).sort_values("Projected Wins", ascending=False)

        # ------------------------------------------
        # VISUALS (Bar Chart with Labels)
        # ------------------------------------------
        st.write("### Projected Win Totals")
        
        fig = px.bar(
            df,
            x="Team",
            y="Projected Wins",
            text="Projected Wins",
            title=f"{conf_name} Projected Win Totals",
            color="Δ Wins",
            color_continuous_scale="Bluered",
            hover_data=["Current Wins", "Projected Losses", "Δ Wins"]
        )
        
        # Center labels inside bars and format nicely
        fig.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside"
        )
        fig.update_layout(
            yaxis_title="Projected Wins",
            xaxis_title="Team",
            showlegend=False,
            uniformtext_minsize=8,
            uniformtext_mode="hide",
            margin=dict(t=60, b=60, l=40, r=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)

