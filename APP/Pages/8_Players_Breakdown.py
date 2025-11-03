# APP/Pages/8_Player.py
# =============================================================================
# Player Dashboard — Season Averages + Share-of-Team "Wheel"
# =============================================================================
import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import List, Optional, Dict, Any
import matplotlib.pyplot as plt

st.set_page_config(page_title="Players", layout="wide")
st.title("Players — Averages & Share of Team")

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
BASE_DIR    = "Data/26_March_Madness_Databook"
PLAYER_PATH = os.path.join(BASE_DIR, "Player Value-Table 1.csv")

# Which stats to consider for averages and the share wheel
STAT_CANDIDATES: Dict[str, List[str]] = {
    "MIN":   ["MIN", "Min", "Minutes"],
    "PTS":   ["PTS", "Points", "Point"],
    "FGM":   ["FGM", "FG Made"],
    "FGA":   ["FGA", "FG Att"],
    "3PM":   ["3PTM", "3PM", "FG3M", "3FGM", "3PT Made"],
    "3PA":   ["3PTA", "3PA", "FG3A", "3FGA", "3PT Att"],
    "FTM":   ["FTM", "FT Made"],
    "FTA":   ["FTA", "FT Att"],
    "OREB":  ["OReb", "OREB", "Off Reb", "Offensive Rebounds"],
    "DREB":  ["DReb", "DREB", "Def Reb", "Defensive Rebounds"],
    "REB":   ["Reb", "REB", "Rebounds"],
    "AST":   ["AST", "Assists"],
    "STL":   ["STL", "Steals"],
    "TOV":   ["TO", "TOV", "Turnovers"],
    "PF":    ["PF", "Fouls"],
}

# Columns that identify player/game/team context
PLAYER_COL_CANDS = ["Player Name", "Player", "Athlete", "Name"]
TEAM_COL_CANDS   = ["Team", "Teams"]
OPP_COL_CANDS    = ["Opponent", "Opp", "Opp Team"]
DATE_COL_CANDS   = ["Date", "Game Date"]
TOP7_COL_CANDS   = ["Top 7", "Top7", "TOP_7", "Is Top 7", "Top Seven"]

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        return None
    try:
        df = pd.read_csv(path, encoding="latin1")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error reading {path}: {e}")
        return None

def find_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    # be forgiving on case
    lower_map = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

def find_first_match(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    # Return the first column name present from a list of aliases (case-insensitive)
    col = find_col(df, names)
    return col

def to_num(s, default=np.nan):
    try:
        v = float(pd.to_numeric(s, errors="coerce"))
        return v if np.isfinite(v) else default
    except Exception:
        return default

def truthy(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().upper()
    if s in ("1","TRUE","YES","Y","T"):
        return True
    try:
        return float(s) > 0
    except Exception:
        return False

def safe_group_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().any():
        return float(vals.mean())
    return np.nan

def build_game_id(row: pd.Series, team_col: str, opp_col: Optional[str], date_col: Optional[str]) -> str:
    team = str(row.get(team_col, "")).strip().lower()
    opp  = str(row.get(opp_col, "")).strip().lower() if opp_col else ""
    dstr = str(row.get(date_col, "")).strip()
    return f"{dstr}|{team}|{opp}"

# -----------------------------------------------------------------------------
# LOAD
# -----------------------------------------------------------------------------
df = load_csv(PLAYER_PATH)
if df is None:
    st.stop()

# Identify key columns
PLAYER_COL = find_col(df, PLAYER_COL_CANDS)
TEAM_COL   = find_col(df, TEAM_COL_CANDS)
OPP_COL    = find_col(df, OPP_COL_CANDS)
DATE_COL   = find_col(df, DATE_COL_CANDS)
TOP7_COL   = find_col(df, TOP7_COL_CANDS)

if PLAYER_COL is None or TEAM_COL is None:
    st.error("Could not locate Player and Team columns in Player Value CSV.")
    st.stop()

# Parse a numeric copy for all stat candidates we actually find
resolved_stat_cols: Dict[str, str] = {}
for std_name, aliases in STAT_CANDIDATES.items():
    col = find_first_match(df, aliases)
    if col is not None:
        resolved_stat_cols[std_name] = col

if "REB" not in resolved_stat_cols:
    # if no explicit REB, try OREB + DREB later for a computed rebound
    pass

# Build helper fields
if DATE_COL is not None:
    df["_date_parsed"] = pd.to_datetime(df[DATE_COL].astype(str), errors="coerce", format="%m/%d/%Y")
else:
    df["_date_parsed"] = pd.NaT

df["_game_id"] = df.apply(lambda r: build_game_id(r, TEAM_COL, OPP_COL, DATE_COL), axis=1)
df["_Top7"] = df[TOP7_COL].apply(truthy) if TOP7_COL else False

# Coerce numeric copies for stats we care about
for std_name, col in resolved_stat_cols.items():
    df[f"__{std_name}"] = pd.to_numeric(df[col], errors="coerce")

# Optional computed REB if missing
if "REB" not in resolved_stat_cols and ("__OREB" in df.columns and "__DREB" in df.columns):
    df["__REB"] = df["__OREB"].fillna(0) + df["__DREB"].fillna(0)
    resolved_stat_cols["REB"] = "__REB"  # mark as synthetic
else:
    # remap to already prefixed numeric column
    for k, c in list(resolved_stat_cols.items()):
        if not c.startswith("__"):
            resolved_stat_cols[k] = f"__{k}" if f"__{k}" in df.columns else c

# -----------------------------------------------------------------------------
# SIDEBAR — TEAM / PLAYER SELECTORS
# -----------------------------------------------------------------------------
st.sidebar.header("Filters")

teams = sorted(df[TEAM_COL].dropna().astype(str).unique().tolist())
team_pick = st.sidebar.selectbox("Team", teams, index=0)

top7_only = st.sidebar.checkbox("Show only Top-7 players", value=True)

team_df = df[df[TEAM_COL].astype(str) == str(team_pick)]
if top7_only and TOP7_COL:
    team_df = team_df[team_df["_Top7"] == True]

players = sorted(team_df[PLAYER_COL].dropna().astype(str).unique().tolist())
if not players:
    st.info("No players found for this selection.")
    st.stop()

player_pick = st.sidebar.selectbox("Player", players, index=0)

# Wheel stat choices
default_wheel = ["PTS", "FGM", "FGA", "3PM", "3PA", "FTM", "FTA", "REB", "AST", "STL", "TOV"]
wheel_stats = st.sidebar.multiselect(
    "Wheel stats (player share of team)",
    options=[k for k in resolved_stat_cols.keys()],
    default=[k for k in default_wheel if k in resolved_stat_cols]
)

# -----------------------------------------------------------------------------
# SEASON AVERAGES — PLAYER
# -----------------------------------------------------------------------------
player_rows = df[(df[TEAM_COL].astype(str) == str(team_pick)) & (df[PLAYER_COL].astype(str) == str(player_pick))]

if player_rows.empty:
    st.info("No rows for this player/team.")
    st.stop()

# Per-game averages = simple mean across games
player_avgs: Dict[str, float] = {}
for k, colname in resolved_stat_cols.items():
    col = colname if colname.startswith("__") else f"__{k}"
    if col not in df.columns:
        continue
    player_avgs[k] = safe_group_mean(player_rows[col])

# -----------------------------------------------------------------------------
# SEASON AVERAGES — TEAM (reconstructed by summing per game, then averaging)
# -----------------------------------------------------------------------------
team_rows = df[df[TEAM_COL].astype(str) == str(team_pick)].copy()

# Sum each game over all players, then average those game totals
team_game_sums = (
    team_rows
    .groupby("_game_id")
    .agg({ (col if col.startswith("__") else f"__{k}"): "sum"
           for k, col in resolved_stat_cols.items()
         })
)

team_avgs: Dict[str, float] = {}
for k, colname in resolved_stat_cols.items():
    col = colname if colname.startswith("__") else f"__{k}"
    if col in team_game_sums.columns:
        team_avgs[k] = float(team_game_sums[col].mean())
    else:
        team_avgs[k] = np.nan

# Guard against 0/NaN for share calc
def pct_share(player_v: float, team_v: float) -> float:
    if not np.isfinite(player_v) or not np.isfinite(team_v) or team_v <= 0:
        return np.nan
    return float(player_v / team_v * 100.0)

share_rows = []
for k in resolved_stat_cols.keys():
    pv = player_avgs.get(k, np.nan)
    tv = team_avgs.get(k, np.nan)
    share_rows.append((k, pv, tv, pct_share(pv, tv)))

summary_df = pd.DataFrame(share_rows, columns=["Stat", "Player Avg", "Team Avg", "% of Team"])
summary_df_display = summary_df.copy()
summary_df_display["Player Avg"] = summary_df_display["Player Avg"].round(2)
summary_df_display["Team Avg"]   = summary_df_display["Team Avg"].round(2)
summary_df_display["% of Team"]  = summary_df_display["% of Team"].round(1)

# -----------------------------------------------------------------------------
# LAYOUT
# -----------------------------------------------------------------------------
c1, c2 = st.columns([1,1])
with c1:
    st.subheader(f"{player_pick} — Season Averages (per game)")
    st.dataframe(summary_df_display, use_container_width=True)

with c2:
    st.subheader("Share of Team — Wheel")
    if wheel_stats:
        # Build pie parts
        labels, values = [], []
        for k in wheel_stats:
            pv = player_avgs.get(k, np.nan)
            tv = team_avgs.get(k, np.nan)
            pct = pct_share(pv, tv)
            if np.isfinite(pct) and pct > 0:
                labels.append(k)
                values.append(pct)
        if values:
            fig, ax = plt.subplots(figsize=(4.5, 4.5))
            wedges, texts, autotexts = ax.pie(
                values,
                labels=[f"{lab} ({v:.1f}%)" for lab, v in zip(labels, values)],
                autopct=None,
                startangle=90,
            )
            ax.axis('equal')
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("No valid stats selected for wheel (team averages may be 0/NaN).")

st.markdown("---")

# Extra: quick context cards
left, right = st.columns([1,1])
with left:
    top7_flag = bool(player_rows["_Top7"].any()) if TOP7_COL else False
    st.markdown(f"**Team:** {team_pick}")
    st.markdown(f"**Top-7:** {'Yes' if top7_flag else 'No'}")
with right:
    # Quick efficiency-style derived stats if available
    extras = []
    if all(x in player_avgs for x in ("FGM","FGA")) and player_avgs["FGA"] and np.isfinite(player_avgs["FGA"]):
        extras.append(f"FG%: {100*player_avgs['FGM']/max(1e-9,player_avgs['FGA']):.1f}%")
    if all(x in player_avgs for x in ("3PM","3PA")) and player_avgs["3PA"] and np.isfinite(player_avgs["3PA"]):
        extras.append(f"3P%: {100*player_avgs['3PM']/max(1e-9,player_avgs['3PA']):.1f}%")
    if all(x in player_avgs for x in ("FTM","FTA")) and player_avgs["FTA"] and np.isfinite(player_avgs["FTA"]):
        extras.append(f"FT%: {100*player_avgs['FTM']/max(1e-9,player_avgs['FTA']):.1f}%")
    if "MIN" in player_avgs and np.isfinite(player_avgs["MIN"]):
        extras.append(f"MIN: {player_avgs['MIN']:.1f}")
    st.markdown("**Quick Glance:** " + (", ".join(extras) if extras else "—"))

# -----------------------------------------------------------------------------
# DOWNLOADS
# -----------------------------------------------------------------------------
st.markdown("---")
csv_name = f"{team_pick}_{player_pick}_averages_and_share.csv".replace(" ", "_")
st.download_button(
    "Download player averages & share CSV",
    data=summary_df_display.to_csv(index=False).encode("utf-8"),
    file_name=csv_name,
    mime="text/csv"
)

