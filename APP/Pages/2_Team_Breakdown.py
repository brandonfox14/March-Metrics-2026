import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# -----------------------
# Load Data
# -----------------------
@st.cache_data
def load_data():
    return pd.read_csv("Data/26_March_Madness_Databook/All_Stats-THE_TABLE.csv", encoding="latin1")

df = load_data()

# -----------------------
# Helper Functions
# -----------------------
def format_value(key_or_label, val):
    if pd.isna(val):
        return "N/A"
    try:
        v = float(val)
    except Exception:
        return str(val)
    if ("PERC" in str(key_or_label).upper()) or ("%" in str(key_or_label)):
        return f"{v:.1%}" if v <= 1 else f"{v:.1f}%"
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.1f}"

def format_rank(val):
    if pd.isna(val):
        return "N/A"
    try:
        return int(float(val))
    except Exception:
        return val

# Rank mapping
rank_overrides = {
    "Points": "Points_RANK",
    "FG_PERC": "FG_PERC_Rank",
    "FGM/G": "FGM/G_Rank",
    "FG3_PERC": "FG3_PERC_Rank",
    "FT_PERC": "FT_PERC_Rank",
    "Off_eff": "Off_eff_rank",
    "Offensive efficiency hybrid": "Off_eff_hybrid_rank",
    "Def_efficiency hybrid": "Def_eff_hybrid_rank",
    "OReb": "OReb Rank",
    "Rebound Rate": "Rebound Rate Rank",
    "AST": "AST Rank",
    "TO": "TO Rank",
    "STL": "STL Rank",
    "PF": "PF_Rank",
    "OPP_PPG": "OPP_PPG_RANK",
    "OPP_FG_PERC": "OPP_FG_PERC_Rank",
    "OPP_FG3_PERC": "OPP_FG3_PERC_Rank",
    "Coach Value": "Coach Value Rank",
    "Culture (coach record at school)": "Culture_rank",
    "Historical Value": "Historical Value Rank"
}

def get_rank_col(key):
    return rank_overrides.get(key)

def robust_normalize(df_section):
    out = pd.DataFrame(index=df_section.index, columns=df_section.columns, dtype=float)
    for c in df_section.columns:
        col = pd.to_numeric(df_section[c], errors="coerce")
        if col.dropna().empty:
            out[c] = 0.5
            continue
        mn, mx = col.min(), col.max()
        out[c] = 0.5 if mx == mn else (col - mn) / (mx - mn)
    return out

# -----------------------
# Team Selection
# -----------------------
teams_sorted = sorted(df["Teams"].dropna().unique())
selected_team = st.selectbox("Select a Team", teams_sorted)
team_data = df[df["Teams"] == selected_team].iloc[0]
team_conf = team_data.get("Conference", None)

# -----------------------
# Chart Builder
# -----------------------
def build_section_chart(section_cols, section_title):
    st.header(f"{selected_team} {section_title}")

    missing = [k for k in section_cols.keys() if k not in df.columns]
    if missing:
        st.warning(f"Missing columns for '{section_title}': {missing}")
        return

    for key, label in section_cols.items():
        col1, col2, col3 = st.columns([3, 2, 3])
        with col1:
            st.markdown(f"**{label}**")
        with col2:
            st.write(format_value(key, team_data.get(key, float('nan'))))
        with col3:
            rank_col = get_rank_col(key)
            rank_val = team_data.get(rank_col, pd.NA) if rank_col else "N/A"
            st.write(format_rank(rank_val))

    # Normalization and comparison
    stat_keys = [k for k in section_cols.keys() if k in df.columns]
    section_df = df[stat_keys].apply(pd.to_numeric, errors="coerce")
    normalized = robust_normalize(section_df)

    team_norm = normalized.loc[df["Teams"] == selected_team].iloc[0].tolist()
    league_norm = normalized.mean(skipna=True).tolist()
    conf_norm = normalized[df["Conference"] == team_conf].mean(skipna=True).tolist() if team_conf else None

    # Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(section_cols.values()), y=team_norm,
                             mode="lines+markers", name=selected_team))
    if conf_norm is not None:
        fig.add_trace(go.Scatter(x=list(section_cols.values()), y=conf_norm,
                                 mode="lines+markers", name=f"{team_conf} Avg", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=list(section_cols.values()), y=league_norm,
                             mode="lines+markers", name="League Avg", line=dict(dash="dot")))

    fig.update_layout(title=f"{section_title} Comparison (Normalized)",
                      yaxis=dict(showticklabels=False, range=[0, 1]),
                      xaxis=dict(tickangle=45),
                      plot_bgcolor="white",
                      margin=dict(t=60, b=120))
    st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Sections
# -----------------------

scoring_cols = {
    "Points": "Points Per Game",
    "PTS_OFF_TURN": "Points Off Turnovers",
    "FST_BREAK": "Fast Break Points",
    "PTS_PAINT": "Points in Paint",
    "Extra Scoring Chances": "Extra Scoring Chances",
    "% of Points from 3": "% of Points from 3",
    "% of shots taken from 3": "% of Shots Taken from 3",
    "CLUTCH_FGPERC": "Clutch FG%",
    "CLUTCH_3FGPERC": "Clutch 3PT%",
    "CLUTCH_FTPERC": "Clutch FT%",
}

offense_cols = {
    "FG_PERC": "Field Goal %",
    "FG3_PERC": "3 Point %",
    "FT_PERC": "Free Throw %",
    "Off_eff": "Offensive Efficiency",
    "Offensive efficiency hybrid": "Off. Efficiency Hybrid",
    "OReb": "Offensive Rebounds",
    "AST": "Assists",
    "TO": "Turnovers",
}

defense_cols = {
    "OPP_PPG": "Opponent Points per Game",
    "OPP_FG_PERC": "Opponent FG%",
    "OPP_FG3_PERC": "Opponent 3PT%",
    "Def_efficiency hybrid": "Defensive Efficiency Hybrid",
    "OPP_OReb": "Opponent Offensive Rebounds",
    "OPP_TO": "Opponent Turnovers",
}

extra_cols = {
    "STL": "Steals",
    "PF": "Personal Fouls",
    "Rebound Rate": "Rebound Rate",
    "Foul Differential": "Foul Differential",
    "SOS_MED": "Median Strength of Schedule",
    "SOS_STDEV": "SOS Variability",
}

coaching_cols = {
    "Coach Value": "Coach Value",
    "Culture (coach record at school)": "Culture",
    "Historical Value": "Historical Value"
}

# -----------------------
# Build Layout
# -----------------------
st.info("ℹ️ The **right column** shows each team's ranking compared to all other teams.")

build_section_chart(scoring_cols, "Scoring Statistics")
build_section_chart(offense_cols, "Offensive Statistics")
build_section_chart(defense_cols, "Defensive Statistics")
build_section_chart(extra_cols, "Extra Statistics")

if any(c in df.columns for c in coaching_cols.keys()):
    build_section_chart(coaching_cols, "Coaching & Culture Statistics")
