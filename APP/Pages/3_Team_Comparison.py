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
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# -----------------------
# Rank mapping (adjustable)
# -----------------------
rank_overrides = {
    # scoring first (per user preference)
    "Extra Scoring Chances": "Extra Scoring Chances Rank",
    "PTS_OFF_TURN": "PTS_OFF_TURN_RANK",
    "FST_BREAK": "FST_BREAK_RANK",
    "PTS_PAINT": "PTS_PAINT_RANK",
    "% of Points from 3": "% of Points from 3_RANK",
    "% of shots taken from 3": "% of shots taken from 3_RANK",

    # offense
    "Points": "Points_RANK",
    "FG_PERC": "FG_PERC_Rank",
    "FGM/G": "FGM/G_Rank",
    "FG3_PERC": "FG3_PERC_Rank",
    "FG3M/G": "FG3M/G_Rank",
    "FT_PERC": "FT_PERC_Rank",
    "FTM/G": "FTM/G_RANK",

    # defense
    "OPP_PPG": "OPP_PPG_RANK",
    "OPP_FG_PERC": "OPP_FG_PERC_Rank",
    "OPP_FGM/G": "OPP_FGM/G_Rank",
    "OPP_FG3_PERC": "OPP_FG3_PERC_Rank",
    "OPP_FG3M/G": "OPP_FG3M/G_Rank",
    "OPP_% of Points from 3": "OPP_% of Points from 3 rank",
    "OPP_% of shots taken from 3": "OPP_% of shots taken from 3 Rank",
    "OPP_OReb": "OPP_OReb_RANK",

    # extras
    "OReb": "OReb Rank",
    "OReb chances": "OReb chances Rank",
    "DReb": "DReb Rank",
    "Rebounds": "Rebounds Rank",
    "Rebound Rate": "Rebound Rate Rank",
    "AST": "AST Rank",
    "AST/FGM": "AST/FGM Rank",
    "TO": "TO Rank",
    "STL": "STL Rank",
    "PF": "PF_Rank",
    "Foul Differential": "Foul Differential Rank",
}

def get_rank_col(key: str):
    return rank_overrides.get(key)

# -----------------------
# Stat groups
# -----------------------
stat_groups = {
    "Scoring": [
        "Extra Scoring Chances", "PTS_OFF_TURN", "FST_BREAK", "PTS_PAINT",
        "% of Points from 3", "% of shots taken from 3"
    ],
    "Offense": [
        "Points","FG_PERC","FGM/G","FG3_PERC","FG3M/G","FT_PERC","FTM/G"
    ],
    "Defense": [
        "OPP_PPG","OPP_FG_PERC","OPP_FGM/G","OPP_FG3_PERC","OPP_FG3M/G",
        "OPP_% of Points from 3","OPP_% of shots taken from 3","OPP_OReb"
    ],
    "Extra Statistical Values": [
        "OReb","OReb chances","DReb","Rebounds","Rebound Rate",
        "AST","AST/FGM","TO","STL","PF","Foul Differential"
    ],
}

# -----------------------
# Helper functions
# -----------------------
def safe_format_value(col_key, val):
    if pd.isna(val):
        return "N/A"
    try:
        v = float(val)
    except Exception:
        return str(val)
    if ("PERC" in str(col_key).upper()) or ("%" in str(col_key)):
        return f"{v:.1%}" if v <= 1 else f"{v:.1f}%"
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.1f}"

def color_by_rank(rank):
    if pd.isna(rank):
        return "rgba(200,200,200,0.6)"
    try:
        r = int(rank)
    except Exception:
        return "rgba(200,200,200,0.6)"
    if r > 200:
        return "rgba(255,140,120,0.8)"
    elif 151 <= r <= 200:
        return "rgba(190,190,190,0.8)"
    else:
        green_val = int(70 + (150 - r) * 1.2)
        green_val = max(70, min(255, green_val))
        return f"rgba(60,{green_val},60,0.85)"

def normalize_stat(val, stat_col):
    if stat_col not in df.columns:
        return 0.5
    col = pd.to_numeric(df[stat_col], errors="coerce")
    if col.dropna().empty:
        return 0.5
    mn, mx = col.min(skipna=True), col.max(skipna=True)
    if pd.isna(val) or mn == mx:
        return 0.5
    try:
        return float((val - mn) / (mx - mn))
    except Exception:
        return 0.5

# -----------------------
# Team selectors
# -----------------------
teams_sorted = sorted(df["Teams"].dropna().unique().tolist())
default_a, default_b = teams_sorted[0], teams_sorted[1] if len(teams_sorted) > 1 else teams_sorted[0]

if "Conference" in df.columns and "Games Dropping D2" in df.columns:
    try:
        sec_team = (
            df[df["Conference"].str.upper() == "SEC"]
            .sort_values("Games Dropping D2", ascending=False)
            .iloc[0]["Teams"]
        )
        big10_team = (
            df[df["Conference"].str.upper().isin(["BIG TEN", "B1G"])]
            .sort_values("Games Dropping D2", ascending=False)
            .iloc[0]["Teams"]
        )
        if pd.notna(sec_team): default_a = sec_team
        if pd.notna(big10_team): default_b = big10_team
    except Exception:
        pass

col1, col2 = st.columns(2)
with col1:
    team_a = st.selectbox("Select Left Team", teams_sorted, index=teams_sorted.index(default_a))
with col2:
    team_b = st.selectbox("Select Right Team", teams_sorted, index=teams_sorted.index(default_b))

team_a_data = df[df["Teams"] == team_a].iloc[0]
team_b_data = df[df["Teams"] == team_b].iloc[0]

# -----------------------
# Team comparison bars
# -----------------------
st.subheader("Team Comparison: Stats")
for group_name, stats in stat_groups.items():
    st.markdown(f"### {group_name}")
    for stat in stats:
        if stat not in df.columns:
            st.markdown(f"*Note: '{stat}' column missing from dataset — skipped.*")
            continue

        rank_col = get_rank_col(stat)
        rank_a = team_a_data.get(rank_col, np.nan) if rank_col else np.nan
        rank_b = team_b_data.get(rank_col, np.nan) if rank_col else np.nan

        val_a = team_a_data.get(stat, np.nan)
        val_b = team_b_data.get(stat, np.nan)

        color_a = color_by_rank(rank_a)
        color_b = color_by_rank(rank_b)

        left_col, center_col, right_col = st.columns([4, 2, 4])
        with left_col:
            st.markdown(
                f"<div style='display:flex; justify-content:flex-end; align-items:center;'>"
                f"<div style='width:60%; background:{color_a}; padding:6px; border-radius:6px; "
                f"text-align:right;'>{safe_format_value(stat, val_a)}</div>"
                f"</div>", unsafe_allow_html=True)
        with center_col:
            st.markdown(f"**{stat}**")
        with right_col:
            st.markdown(
                f"<div style='display:flex; justify-content:flex-start; align-items:center;'>"
                f"<div style='width:60%; background:{color_b}; padding:6px; border-radius:6px; "
                f"text-align:left;'>{safe_format_value(stat, val_b)}</div>"
                f"</div>", unsafe_allow_html=True)

# -----------------------
# Radar chart comparison
# -----------------------
st.subheader("Team Radar: Average Rankings")

def avg_rank_for_keys(team_data, keys):
    ranks = []
    for k in keys:
        rc = get_rank_col(k)
        if rc and rc in df.columns:
            v = team_data.get(rc, np.nan)
            if not pd.isna(v):
                try:
                    ranks.append(float(v))
                except Exception:
                    pass
    return float(np.mean(ranks)) if ranks else np.nan

def overall_avg_rank(team_data):
    all_keys = [s for group in stat_groups.values() for s in group]
    ranks = []
    for k in all_keys:
        rc = get_rank_col(k)
        if rc and rc in df.columns:
            v = team_data.get(rc, np.nan)
            if not pd.isna(v):
                try:
                    ranks.append(float(v))
                except Exception:
                    pass
    return float(np.mean(ranks)) if ranks else np.nan

radar_categories = list(stat_groups.keys())
overall_a = overall_avg_rank(team_a_data)
overall_b = overall_avg_rank(team_b_data)

ranks_a = [avg_rank_for_keys(team_a_data, stat_groups[g]) for g in radar_categories]
ranks_b = [avg_rank_for_keys(team_b_data, stat_groups[g]) for g in radar_categories]

fig = go.Figure()
fig.add_trace(go.Scatterpolar(
    r=ranks_a,
    theta=radar_categories,
    fill='toself',
    name=team_a
))
fig.add_trace(go.Scatterpolar(
    r=ranks_b,
    theta=radar_categories,
    fill='toself',
    name=team_b
))

# ✅ Safe handling for early-season NAs
all_rank_cols = [c for c in rank_overrides.values() if c in df.columns]
if all_rank_cols:
    rank_data = df[all_rank_cols].apply(pd.to_numeric, errors="coerce")
    if rank_data.notna().any().any():
        max_rank_observed = int(np.nanmax(rank_data.max(skipna=True)))
    else:
        max_rank_observed = 365
else:
    max_rank_observed = 365

max_rank = max(365, max_rank_observed)

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[max_rank, 1],
            tickvals=[50,100,150,200,250,300,350]
        )
    ),
    showlegend=True,
    title="Average Category Rankings (1 = Best, outer circle)"
)

st.plotly_chart(fig, use_container_width=True)
