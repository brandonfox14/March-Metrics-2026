# APP/Pages/8_Player.py
# =============================================================================
# Player (8) — Team → Top 7 player breakdown + "Other" + pie shares per stat
# =============================================================================
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from typing import List, Optional, Dict, Tuple, Any

# ---------------------------------------
# Config
# ---------------------------------------
BASE_DIR = "Data/26_March_Madness_Databook"
PLAYER_PATH = os.path.join(BASE_DIR, "Player Value-Table 1.csv")

st.set_page_config(page_title="Players — Top 7 Breakdown", layout="wide")
st.title("Players — Top 7 Breakdown")

# ---------------------------------------
# Helpers
# ---------------------------------------
def dedupe_columns(cols: List[str]) -> List[str]:
    """Make column names unique if file has duplicates."""
    seen = {}
    out = []
    for c in cols:
        cc = str(c).strip()
        if cc not in seen:
            seen[cc] = 0
            out.append(cc)
        else:
            seen[cc] += 1
            out.append(f"{cc}__dup{seen[cc]}")
    return out

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def parse_bool(x: Any) -> bool:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return False
    s = str(x).strip().upper()
    if s in ("1", "TRUE", "YES", "Y", "TOP7", "T", "TOP 7"):
        return True
    try:
        return float(s) > 0
    except Exception:
        return False

def is_numeric_col(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    # text that looks numeric
    try:
        pd.to_numeric(series.dropna().astype(str).str.replace(",", ""), errors="raise")
        return True
    except Exception:
        return False

@st.cache_data
def load_player_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="latin1")
    except Exception:
        df = pd.read_csv(path)  # fallback
    df.columns = dedupe_columns(df.columns.tolist())
    df.columns = [c.strip() for c in df.columns]
    return df

def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def pick_stat_families(numeric_cols: List[str]) -> Dict[str, List[str]]:
    """
    Build a few sensible families for quick browsing.
    Only include columns that actually exist.
    """
    def present(cands: List[str]) -> List[str]:
        return [c for c in cands if c in numeric_cols]

    families = {
        "Scoring": present(["Points", "FGM", "FGA", "FG3sM", "FG3sA", "FTM", "FTA"]),
        "Shooting %": present(["FG_PERC", "FG3_PERC", "FT_PERC"]),
        "Rebounding": present(["OReb", "DReb", "Rebounds"]),
        "Playmaking": present(["AST", "TO"]),
        "Defense": present(["STL", "PF"]),
        "Misc": present(["MIN", "PTS_OFF_TURN", "FST_BREAK", "PTS_PAINT"])
    }
    # prune empties
    return {k: v for k, v in families.items() if v}

def sanitize_stat_choice(stat: str, numeric_cols: List[str]) -> str:
    if stat in numeric_cols:
        return stat
    # Heuristic mapping if users have slight column naming variation
    aliases = {
        "3PTM": "FG3sM",
        "3PTA": "FG3sA",
        "OREB": "OReb",
        "DREB": "DReb",
        "REB": "Rebounds",
        "ASTS": "AST",
        "STLS": "STL",
        "FGA/G": "FGA/G",  # in case exists
    }
    return aliases.get(stat, numeric_cols[0] if numeric_cols else stat)

# ---------------------------------------
# Load data
# ---------------------------------------
df = load_player_csv(PLAYER_PATH)
if df is None:
    st.error(f"Player file not found: {PLAYER_PATH}")
    st.stop()

# likely columns
TEAM_COL = find_col(df, ["Team", "Teams"])
PLAYER_COL = find_col(df, ["Player", "Player Name", "Name"])
TOP7_COL = find_col(df, ["Top 7", "Top7", "TOP7", "Starter Top7", "Top_7"])

if TEAM_COL is None or PLAYER_COL is None:
    st.error("Could not detect Team and Player columns in Player Value-Table 1.csv")
    st.stop()

# mark Top7 boolean (fallback: use "Starter" if Top7 missing)
if TOP7_COL is None:
    STARTER_COL = find_col(df, ["Starter", "Bench/Starter", "Start", "IsStarter"])
    if STARTER_COL is not None:
        df["__Top7"] = df[STARTER_COL].apply(parse_bool)
        TOP7_COL = "__Top7"
    else:
        df["__Top7"] = False
        TOP7_COL = "__Top7"
else:
    df["__Top7"] = df[TOP7_COL].apply(parse_bool)
    TOP7_COL = "__Top7"

# identify numeric stats
numeric_cols = [c for c in df.columns if is_numeric_col(df[c])]
df = coerce_numeric(df, numeric_cols)

# ---------------------------------------
# Sidebar — Team & Stat Controls
# ---------------------------------------
teams = sorted(list({str(x) for x in df[TEAM_COL].dropna().unique()}))
team_selected = st.sidebar.selectbox("Select Team", teams, index=0 if teams else None)

families = pick_stat_families(numeric_cols)
if not families:
    st.warning("No numeric stat families detected; falling back to all numeric columns.")
    families = {"All": [c for c in numeric_cols if c not in ("Wins", "Losses")]}

family_names = list(families.keys())
family_choice = st.sidebar.selectbox("Choose Stat Family", family_names)
stat_options = families[family_choice]
default_stat = sanitize_stat_choice("Points", stat_options) if "Points" in stat_options else stat_options[0]
stat_choice = st.sidebar.selectbox("Stat for Pie", stat_options, index=stat_options.index(default_stat))

# ---------------------------------------
# Slice team, aggregate per-player averages
# ---------------------------------------
team_df = df.loc[df[TEAM_COL].astype(str) == str(team_selected)].copy()
if team_df.empty:
    st.info("No rows for the selected team.")
    st.stop()

# group by player → mean averages for all numeric stats + count of games
grouped = team_df.groupby(PLAYER_COL)
player_avg = grouped[numeric_cols].mean(numeric_only=True)
player_games = grouped.size().rename("Games")

player_out = player_avg.join(player_games)
player_out["Top7"] = grouped[TOP7_COL].apply(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[-1]).astype(bool)

# split Top7 vs Others
top7_df = player_out[player_out["Top7"]].copy()
others_df = player_out[~player_out["Top7"]].copy()

# build "Other" aggregate row (mean across non-top7 players OR sum? → use mean for fairness to “averages” comparison;
# for pie share (team share), we will compute slice = value for each player and "Other" = sum of others).
other_row = None
if not others_df.empty:
    other_row = others_df.mean(numeric_only=True)
    # For “Games”, use sum of games for others (more informative)
    other_row["Games"] = others_df["Games"].sum()
    other_row["Top7"] = False

# Make display tables (Top7 players individually; Others collapsed below)
st.markdown("---")
st.subheader(f"{team_selected} — Top 7 Players (per-game averages)")

def nice_table(df_in: pd.DataFrame, keep_cols: Optional[List[str]] = None) -> pd.DataFrame:
    tbl = df_in.reset_index().copy()
    # Small, focused default columns if available
    defaults = ["Games", "MIN", "Points", "FGM", "FGA", "FG_PERC", "FG3sM", "FG3sA", "FG3_PERC",
                "FTM", "FTA", "FT_PERC", "OReb", "DReb", "Rebounds", "AST", "TO", "STL", "PF"]
    ordered = [c for c in defaults if c in tbl.columns]
    if keep_cols:
        ordered = [c for c in keep_cols if c in tbl.columns]
    rest = [c for c in tbl.columns if c not in ordered + [PLAYER_COL]]
    tbl = tbl[[PLAYER_COL] + ordered + rest]
    # Round numerics for readability
    for c in tbl.columns:
        if c == PLAYER_COL: 
            continue
        if pd.api.types.is_numeric_dtype(tbl[c]):
            tbl[c] = tbl[c].round(2)
    return tbl

if top7_df.empty:
    st.info("No Top 7 flagged players for this team.")
else:
    st.dataframe(nice_table(top7_df), use_container_width=True)

# Show collapsed "Other" especially for shares
st.markdown("#### Other (Non-Top-7)")
if other_row is None:
    st.write("No non-Top-7 players on record.")
else:
    other_tbl = pd.DataFrame([other_row])
    other_tbl.insert(0, PLAYER_COL, "Other")
    st.dataframe(nice_table(other_tbl), use_container_width=True)

# ---------------------------------------
# Pie (wheel) — player shares of team for the chosen stat
# ---------------------------------------
st.markdown("---")
st.subheader(f"Team Share by Player — {stat_choice}")

# compute pie data: each Top7 player as a slice; Others collapsed into one slice
def build_pie_series(stat: str) -> Tuple[pd.DataFrame, float]:
    # Player value = the per-player average of `stat`
    series = []
    if not top7_df.empty:
        for pname, row in top7_df.iterrows():
            val = row.get(stat, np.nan)
            if np.isfinite(val):
                series.append((pname, float(val), True))
    # Others: sum of averages from non-Top7 players (so slice is meaningful)
    if not others_df.empty and stat in others_df.columns:
        others_sum = float(others_df[stat].fillna(0.0).sum())
        if others_sum > 0:
            series.append(("Other", others_sum, False))
    pie_df = pd.DataFrame(series, columns=["Player", "Value", "Top7"])
    total = float(pie_df["Value"].sum()) if not pie_df.empty else 0.0
    return pie_df, total

pie_df, total_val = build_pie_series(stat_choice)

if pie_df.empty or total_val <= 0:
    st.info(f"No positive data for '{stat_choice}' to build a share chart.")
else:
    pie_df["Percent"] = pie_df["Value"] / total_val * 100.0
    fig = px.pie(
        pie_df,
        names="Player",
        values="Value",
        hole=0.35,
        hover_data={"Value": ":.2f", "Percent": ":.1f", "Player": True, "Top7": True},
        title=f"{team_selected} — {stat_choice} Shares (per-game averages)",
    )
    fig.update_traces(textposition="inside", texttemplate="%{label}<br>%{percent:.1%}")
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------
# Multi-stat quick browse
# ---------------------------------------
st.markdown("---")
st.subheader("Quick Stat Browser")
colL, colR = st.columns([2, 3])

with colL:
    st.write("**Top 7 (per-game averages)**")
    show_cols = families[family_choice]  # current family
    top7_family = top7_df[show_cols].round(2) if not top7_df.empty else pd.DataFrame()
    if not top7_df.empty:
        st.dataframe(
            top7_family.reset_index().rename(columns={PLAYER_COL: "Player"}),
            use_container_width=True
        )
    else:
        st.write("No Top 7 to show.")

with colR:
    st.write("**Other (collapsed)**")
    if other_row is not None:
        st.dataframe(pd.DataFrame([other_row[show_cols].round(2)]).assign(Player="Other"), use_container_width=True)
    else:
        st.write("No 'Other' rows.")

# ---------------------------------------
# Download section
# ---------------------------------------
st.markdown("---")
st.subheader("Export")
export_top7 = top7_df.reset_index()
export_top7.insert(0, "Team", team_selected)
export_top7 = export_top7.rename(columns={PLAYER_COL: "Player"})
export_top7_csv = export_top7.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Top-7 Averages (CSV)",
    data=export_top7_csv,
    file_name=f"{team_selected}_top7_player_averages.csv",
    mime="text/csv"
)
