# APP/Pages/8_Player.py
# =========================================================
# Player (8) — Team → Top 7 player averages & "share of team" wheel
# =========================================================
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Optional, Tuple, Dict, Any

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
BASE_DIR = "Data/26_March_Madness_Databook"
PLAYER_PATH = os.path.join(BASE_DIR, "Player Value-Table 1.csv")  # <-- you said this is the file

st.set_page_config(page_title="Players — Top 7 & Team Share", layout="wide")
st.title("Players — Top 7 & Team Share")

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        return None
    try:
        df = pd.read_csv(path, encoding="latin1")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Failed to read {path}: {e}")
        return None

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def truthy(x: Any) -> bool:
    if pd.isna(x): return False
    s = str(x).strip().upper()
    if s in ("1", "TRUE", "YES", "Y"): return True
    try:
        return float(s) > 0
    except Exception:
        return False

def to_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(s, errors="coerce")
    return out.fillna(default)

def pick_top7_for_team(df_team: pd.DataFrame, col_top7: Optional[str], player_col: str, minutes_col: Optional[str]) -> List[str]:
    """
    Returns a list of player names (len ≤ 7) for the team's 'Top 7'.
    Priority:
      1) Use 'Top 7' flag if present
      2) Else choose 7 highest average minutes (if minutes column exists)
      3) Else choose 7 most frequent starters (if 'Starter' column exists)
      4) Else choose players with most games played
    """
    if df_team.empty:
        return []

    # 1) Use explicit Top 7 column if present
    if col_top7 and col_top7 in df_team.columns:
        top7_names = df_team.groupby(player_col)[col_top7].apply(lambda s: s.map(truthy).mean()).sort_values(ascending=False)
        # mean>0 means flagged as Top7 in at least some rows; keep top 7 of those >0
        explicit = top7_names[top7_names > 0].index.tolist()
        if explicit:
            return explicit[:7]

    # 2) Highest average minutes
    if minutes_col and minutes_col in df_team.columns:
        mins = df_team.groupby(player_col)[minutes_col].mean().sort_values(ascending=False)
        return mins.index.tolist()[:7]

    # 3) Most frequent "Starter" if column exists and is categorical-like
    starter_col = find_col(df_team, ["Starter", "STARTER", "Bench/Starter", "BENCH/STARTER"])
    if starter_col:
        starter_rate = df_team.assign(__starter=df_team[starter_col].astype(str).str.strip().str.upper().eq("STARTER").astype(int))
        starter_rate = starter_rate.groupby(player_col)["__starter"].mean().sort_values(ascending=False)
        return starter_rate.index.tolist()[:7]

    # 4) Most games played
    counts = df_team.groupby(player_col).size().sort_values(ascending=False)
    return counts.index.tolist()[:7]

def agg_player_averages(df_team: pd.DataFrame, player_col: str, stat_cols: List[str]) -> pd.DataFrame:
    """
    Per-player per-team averages for provided stat_cols.
    """
    grp = df_team.groupby(player_col)[stat_cols].mean(numeric_only=True)
    grp = grp.reset_index().rename(columns={player_col: "Player"})
    return grp

def build_team_totals_from_players(df_team_avg: pd.DataFrame, stat: str) -> float:
    """
    Team total for 'stat' derived by summing per-player averages.
    (Works with season averages because team total = sum of player contributions.)
    """
    return float(df_team_avg[stat].sum())

def pct(x, total):
    if total <= 0: return 0.0
    return float(x) / float(total)

# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------
player_df = load_csv(PLAYER_PATH)
if player_df is None:
    st.stop()

# ---------------------------------------------------------
# COLUMN MAPS / DETECTION
# ---------------------------------------------------------
TEAM_COL   = find_col(player_df, ["Teams", "Team", "TEAM", "School"])
OPP_COL    = find_col(player_df, ["Opponent", "Opp", "OPP"])
HAN_COL    = find_col(player_df, ["HAN", "Home/Away", "Location", "Loc", "HomeAway"])
CONF_COL   = find_col(player_df, ["Conference", "Conf"])
OPP_CONF_COL = find_col(player_df, ["Opponent Conference", "Opp Conference"])
PLAYER_COL = find_col(player_df, ["Player Name", "Player", "PLAYER"])
TOP7_COL   = find_col(player_df, ["Top 7", "Top7", "TOP7"])
MIN_COL    = find_col(player_df, ["MIN", "Minutes", "Min"])

# Common per-player stat columns found in your sample
CANDIDATE_STATS = [
    "PTS", "Points", "FGM", "FGA", "3PTM", "3PTA", "FTM", "FTA",
    "OReb", "DReb", "Rebounds", "AST", "Assists", "TO", "Turnovers",
    "STL", "Steals", "PF", "Personal Fouls", "MIN"
]
# Normalize to the actual columns present
STAT_COLS = [c for c in CANDIDATE_STATS if c in player_df.columns]

# Also include any obvious “Rank” columns later if you want, but for the wheel we need raw count stats.

missing = []
for must in [TEAM_COL, PLAYER_COL]:
    if must is None:
        missing.append(must)
if missing:
    st.error("Player file must include at least 'Teams' (or Team) and 'Player Name' columns.")
    st.stop()

# Ensure numeric for stat cols
for c in STAT_COLS:
    player_df[c] = pd.to_numeric(player_df[c], errors="coerce")

# ---------------------------------------------------------
# SIDEBAR — SELECTIONS
# ---------------------------------------------------------
st.sidebar.header("Filters")
teams = sorted(player_df[TEAM_COL].dropna().astype(str).unique().tolist())
team_selected = st.sidebar.selectbox("Select Team", teams)

# pick default stat order preference
default_stat = "Points" if "Points" in STAT_COLS else ("PTS" if "PTS" in STAT_COLS else STAT_COLS[0])
stat_selected = st.sidebar.selectbox("Stat for wheel (share of team)", STAT_COLS, index=STAT_COLS.index(default_stat))

include_other = st.sidebar.checkbox("Include 'Other' (non-Top-7) slice in wheel", value=True)

# ---------------------------------------------------------
# TEAM SLICE
# ---------------------------------------------------------
team_mask = (player_df[TEAM_COL].astype(str) == str(team_selected))
team_df = player_df.loc[team_mask].copy()

# Detect Top 7 list
top7_players = pick_top7_for_team(team_df, TOP7_COL, PLAYER_COL, MIN_COL)
if not top7_players:
    st.warning("Could not determine Top 7. Showing the 7 most-used players by fallback logic.")
top7_set = set(top7_players)

# Compute per-player averages for this team
if not STAT_COLS:
    st.error("No numeric player stat columns found to average.")
    st.stop()

team_player_avg = agg_player_averages(team_df, PLAYER_COL, STAT_COLS)
team_player_avg["IsTop7"] = team_player_avg["Player"].isin(top7_set)

# Split Top7 vs Others
df_top7 = team_player_avg[team_player_avg["IsTop7"]].copy()
df_other = team_player_avg[~team_player_avg["IsTop7"]].copy()

# If explicit Top7 list had <7, fill up from best minutes/usage to reach 7
if len(df_top7) < 7 and not df_other.empty:
    need = 7 - len(df_top7)
    # fallback: highest MIN (if exists), else highest of the selected stat
    if MIN_COL and MIN_COL in team_df.columns:
        extra_order = df_other.sort_values(by="MIN" if "MIN" in df_other.columns else stat_selected, ascending=False)
    else:
        extra_order = df_other.sort_values(by=stat_selected, ascending=False)
    take = extra_order.head(need).copy()
    take["IsTop7"] = True
    df_top7 = pd.concat([df_top7, take], ignore_index=True)
    df_other = team_player_avg[~team_player_avg["Player"].isin(df_top7["Player"])].copy()

# Build an "Other" aggregate row (averages sum) for the wheel
other_row = None
if include_other and not df_other.empty:
    sums = df_other[STAT_COLS].sum(numeric_only=True)
    other_row = pd.DataFrame([{"Player": "Other"} | sums.to_dict()])

# ---------------------------------------------------------
# TOP 7 — AVERAGES TABLE
# ---------------------------------------------------------
st.subheader(f"Top 7 — {team_selected}")
cols_to_show = ["Player"] + STAT_COLS
top7_table = df_top7.sort_values(by=stat_selected, ascending=False)[cols_to_show].reset_index(drop=True)
st.dataframe(top7_table, use_container_width=True)

# ---------------------------------------------------------
# WHEEL (PIE) — PLAYER SHARE OF TEAM FOR SELECTED STAT
# ---------------------------------------------------------
st.markdown("---")
st.subheader(f"Share of Team — {stat_selected}")

# Build wheel data = Top7 + optional Other
wheel_df = df_top7[["Player", stat_selected]].copy()
if other_row is not None:
    wheel_df = pd.concat([wheel_df, other_row[["Player", stat_selected]]], ignore_index=True)

# Use non-negative values only
wheel_df[stat_selected] = wheel_df[stat_selected].clip(lower=0)
team_total_for_stat = float(wheel_df[stat_selected].sum())

if team_total_for_stat <= 0:
    st.info(f"No positive values for {stat_selected} to render a wheel.")
else:
    wheel_df["Share"] = wheel_df[stat_selected] / team_total_for_stat
    pie = px.pie(
        wheel_df,
        names="Player",
        values="Share",
        hole=0.35,
    )
    pie.update_traces(textposition="inside", texttemplate="%{label}<br>%{percent:.1%}")
    pie.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(pie, use_container_width=True)

# ---------------------------------------------------------
# TOP-7 vs TEAM — QUICK READ + BAR
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Top-7 vs Team — selected stat")

top7_sum = float(df_top7[stat_selected].clip(lower=0).sum())
team_sum = float(team_player_avg[stat_selected].clip(lower=0).sum())
top7_pct = (top7_sum / team_sum * 100.0) if team_sum > 0 else 0.0

c1, c2 = st.columns([1, 2])
with c1:
    st.metric("Top-7 Contribution", f"{top7_pct:.1f}%")
with c2:
    bar = go.Figure()
    bar.add_bar(name="Top 7", x=["Top-7 vs Team"], y=[top7_sum])
    bar.add_bar(name="Other", x=["Top-7 vs Team"], y=[max(team_sum - top7_sum, 0.0)])
    bar.update_layout(barmode="stack", showlegend=True, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(bar, use_container_width=True)

# ---------------------------------------------------------
# NON-TOP-7 TABLE (optional detail)
# ---------------------------------------------------------
with st.expander("Other (non-Top-7) — per-player averages", expanded=False):
    if df_other.empty:
        st.write("No additional players.")
    else:
        other_table = df_other.sort_values(by=stat_selected, ascending=False)[cols_to_show].reset_index(drop=True)
        st.dataframe(other_table, use_container_width=True)

# ---------------------------------------------------------
# NOTES
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "Notes:\n"
    "- Top 7 first uses the ‘Top 7’ column if present; otherwise falls back to highest average minutes, then starters, then games played.\n"
    "- Averages are per-player over all rows for the selected team. The wheel shows each player’s share of the team’s summed averages for the chosen stat.\n"
    "- The ‘Other’ slice aggregates all non-Top-7 players."
)
