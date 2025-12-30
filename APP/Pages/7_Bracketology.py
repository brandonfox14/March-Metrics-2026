# APP/Pages/7_Bracketology.py
# =========================================================
# Bracketology — AQs + At-larges + Bubble Buckets
# =========================================================
import os
import numpy as np
import pandas as pd
import streamlit as st

BASE = "Data/26_March_Madness_Databook"

ALL_STATS_FILE = os.path.join(BASE, "All_Stats-Table 1.csv")   # adjust name if needed
SOS_FILE       = os.path.join(BASE, "SOS-Table 1.csv")         # adjust name if needed
CONF_FILE      = os.path.join(BASE, "Conferences-Table 1.csv") # adjust name if needed

st.set_page_config(page_title="Bracketology", layout="wide")
st.title("Bracketology")

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # standardize team col if needed
    if "Teams" in df.columns and "Team" not in df.columns:
        df = df.rename(columns={"Teams": "Team"})
    return df

all_stats = load_csv(ALL_STATS_FILE)
sos = load_csv(SOS_FILE)
confs = load_csv(CONF_FILE)

# Defensive sanity
for c in ["Team", "Conference", "Wins", "Losses"]:
    if c not in all_stats.columns:
        st.error(f"Missing required column in All_Stats: {c}")
        st.stop()

# -----------------------------
# Helper: percentile rank (0..1, higher is better)
# -----------------------------
def pct_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = series.copy()
    s = pd.to_numeric(s, errors="coerce")
    r = s.rank(pct=True, ascending=not higher_is_better)
    return r.fillna(r.median())

# -----------------------------
# Build résumé features from SOS (team-level rollups)
# -----------------------------
# Expect SOS has: Team, H/A/N, Road Game, Neutral Site Game, Top 25 Opponent,
# March Madness Opponent, Tournament Caliber Game, Won as Underdog, Lost as Favorite, SM, Line, (SM+Line)^2 etc.
needed_sos_cols = ["Team"]
if "Team" not in sos.columns and "Teams" in sos.columns:
    sos = sos.rename(columns={"Teams": "Team"})

if "Team" not in sos.columns:
    st.warning("SOS file missing Team column; résumé rollups will be limited.")
    sos_team = pd.DataFrame({"Team": all_stats["Team"].unique()})
else:
    sos_work = sos.copy()

    # coerce common numeric cols if present
    numeric_candidates = [
        "Road Game", "Neutral Site Game", "Top 25 Opponent", "March Madness Opponent",
        "Tournament Caliber Game", "Won as Underdog", "Lost as Favorite",
        "Win Game within 8", "Within 10 line", "Line over 20",
        "SOS Median", "SOS Min", "SOS Max", "SOS STDEV", "SOS SUM",
        "(SM+Line)^2", "(SM+Line+10)^2", "Cover Spread", "Volatility"
    ]
    for col in numeric_candidates:
        if col in sos_work.columns:
            sos_work[col] = pd.to_numeric(sos_work[col], errors="coerce").fillna(0)

    agg_map = {}
    for col in ["Road Game", "Neutral Site Game", "Top 25 Opponent", "March Madness Opponent",
                "Tournament Caliber Game", "Won as Underdog", "Lost as Favorite",
                "Win Game within 8", "Within 10 line", "Line over 20", "Cover Spread"]:
        if col in sos_work.columns:
            agg_map[col] = "sum"

    for col in ["(SM+Line)^2", "(SM+Line+10)^2", "Volatility"]:
        if col in sos_work.columns:
            agg_map[col] = "mean"

    # If your SOS table already carries SOS_* on each row, averaging is fine
    for col in ["SOS Median", "SOS Min", "SOS Max", "SOS STDEV", "SOS SUM"]:
        if col in sos_work.columns:
            agg_map[col] = "mean"

    sos_team = sos_work.groupby("Team", as_index=False).agg(agg_map) if agg_map else pd.DataFrame({"Team": all_stats["Team"].unique()})

# -----------------------------
# Merge into team table
# -----------------------------
df = all_stats.merge(sos_team, on="Team", how="left")

# -----------------------------
# Compute two scores:
#   1) Selection Score (IN vs OUT)
#   2) Seed Score (quality once IN)
# -----------------------------
# You can tune weights easily here.

# --- Selection Score Inputs (résumé + trust)
selection_components = {}

# résumé strength
if "SOS SUM" in df.columns:
    selection_components["sos_sum"] = pct_rank(df["SOS SUM"], True)
elif "SOS_SUM" in df.columns:
    selection_components["sos_sum"] = pct_rank(df["SOS_SUM"], True)

# road/neutral opportunity
for col, key in [("Road Game", "road_games"), ("Neutral Site Game", "neutral_games")]:
    if col in df.columns:
        selection_components[key] = pct_rank(df[col], True)

# “good opponents” counts
for col, key in [("Top 25 Opponent", "top25_opp"), ("March Madness Opponent", "mm_opp"), ("Tournament Caliber Game", "tcg")]:
    if col in df.columns:
        selection_components[key] = pct_rank(df[col], True)

# risk / downside
if "Lost as Favorite" in df.columns:
    selection_components["lost_fav"] = pct_rank(df["Lost as Favorite"], False)

# clutch trust / coach value (your edge)
if "CLUTCH_SM" in df.columns:
    selection_components["clutch_sm"] = pct_rank(df["CLUTCH_SM"], True)
if "Sub 0 Clutch Performances" in df.columns:
    selection_components["sub0_clutch"] = pct_rank(df["Sub 0 Clutch Performances"], False)
if "Coach Value" in df.columns:
    selection_components["coach_value"] = pct_rank(df["Coach Value"], True)

# baseline W-L context
df["WIN_PCT"] = pd.to_numeric(df.get("WIN_PERC", (df["Wins"] / (df["Wins"] + df["Losses"]))), errors="coerce").fillna(0)
selection_components["win_pct"] = pct_rank(df["WIN_PCT"], True)

# Weights (tune)
w_sel = {
    "sos_sum": 0.20,
    "road_games": 0.07,
    "neutral_games": 0.03,
    "top25_opp": 0.06,
    "mm_opp": 0.06,
    "tcg": 0.06,
    "lost_fav": 0.08,
    "clutch_sm": 0.14,
    "sub0_clutch": 0.12,
    "coach_value": 0.06,
    "win_pct": 0.12,
}

# Build score
sel_score = np.zeros(len(df))
weight_used = 0.0
for k, w in w_sel.items():
    if k in selection_components:
        sel_score += w * selection_components[k].to_numpy()
        weight_used += w
df["Selection_Score"] = 100 * (sel_score / max(weight_used, 1e-9))

# --- Seed Score Inputs (quality)
seed_components = {}

# Use ranks if you have them (lower rank is better), convert to “higher is better”
def rank_to_quality(rank_series: pd.Series) -> pd.Series:
    r = pd.to_numeric(rank_series, errors="coerce")
    # invert percentile: smaller ranks => higher quality
    return pct_rank(-r, True)

for col, key in [("Off_eff_hybrid_rank", "off_rank"),
                 ("Def_eff_hybrid_rank", "def_rank"),
                 ("Off_eff_rank", "off_rank2"),
                 ("Def_eff_hybrid_rank", "def_rank2")]:
    if col in df.columns:
        seed_components[key] = rank_to_quality(df[col])

if "SM" in df.columns:
    seed_components["sm"] = pct_rank(df["SM"], True)

if "Rebound Rate" in df.columns:
    seed_components["reb_rate"] = pct_rank(df["Rebound Rate"], True)
if "Rebound Rate Rank" in df.columns:
    seed_components["reb_rate_rank"] = rank_to_quality(df["Rebound Rate Rank"])

if "TO Rank" in df.columns:
    seed_components["to_rank"] = rank_to_quality(df["TO Rank"])
if "Volatility" in df.columns:
    seed_components["volatility"] = pct_rank(df["Volatility"], False)  # lower volatility -> better seed stability

w_seed = {
    "off_rank": 0.24,
    "def_rank": 0.24,
    "sm": 0.20,
    "reb_rate": 0.10,
    "to_rank": 0.10,
    "volatility": 0.12,
}
seed_score = np.zeros(len(df))
w_used = 0.0
for k, w in w_seed.items():
    if k in seed_components:
        seed_score += w * seed_components[k].to_numpy()
        w_used += w
df["Seed_Score"] = 100 * (seed_score / max(w_used, 1e-9))

# -----------------------------
# AQ selection UI (one per conference)
# -----------------------------
st.subheader("Automatic Qualifiers (AQs)")

conf_list = sorted(df["Conference"].dropna().unique().tolist())
default_aqs = []

# If confs table already has champs, you can auto-fill here.
# For now, let you pick.
cols = st.columns(4)
aq_map = {}
for i, conf in enumerate(conf_list):
    c = cols[i % 4]
    teams_in_conf = df.loc[df["Conference"] == conf, "Team"].sort_values().tolist()
    if not teams_in_conf:
        continue
    pick = c.selectbox(conf, teams_in_conf, index=0, key=f"aq_{conf}")
    aq_map[conf] = pick

aqs = sorted(set(aq_map.values()))

# -----------------------------
# Build the projected field
# -----------------------------
st.subheader("Projected Field")

FIELD_SIZE = 68
AQ_COUNT = len(aqs)
AT_LARGE_COUNT = FIELD_SIZE - AQ_COUNT

df["Is_AQ"] = df["Team"].isin(aqs)

# pick at-larges by Selection_Score excluding AQs
atl = (
    df.loc[~df["Is_AQ"]]
      .sort_values(["Selection_Score", "Seed_Score"], ascending=False)
      .head(AT_LARGE_COUNT)["Team"]
      .tolist()
)

field = set(aqs) | set(atl)
df["In_Field"] = df["Team"].isin(field)

# Seed line ordering within field by Seed_Score
field_df = df.loc[df["In_Field"]].copy().sort_values("Seed_Score", ascending=False)
field_df["Overall_Rank"] = np.arange(1, len(field_df) + 1)

# Approx seed number: 1..16 (4 teams per seed line for 64 slots; play-in handled below)
# We'll compute a "seed_line" for the 68 field; you can refine later.
field_df["Seed_Line"] = ((field_df["Overall_Rank"] - 1) // 4) + 1
field_df["Seed_Line"] = field_df["Seed_Line"].clip(1, 16)

# Bubble around cutline (at-larges only)
atl_df = df.loc[~df["Is_AQ"]].sort_values(["Selection_Score", "Seed_Score"], ascending=False).copy()
atl_df["ATL_Rank"] = np.arange(1, len(atl_df) + 1)

# Last 4 byes = at-large ranks (AT_LARGE_COUNT-4 .. AT_LARGE_COUNT-1)
# Last 4 in   = at-large ranks (AT_LARGE_COUNT .. AT_LARGE_COUNT+3)
# First 4 out = next 4
# Next 4 out  = next 4
l4b = atl_df.iloc[max(AT_LARGE_COUNT-4, 0):AT_LARGE_COUNT]["Team"].tolist()
l4i = atl_df.iloc[AT_LARGE_COUNT:AT_LARGE_COUNT+4]["Team"].tolist()
f4o = atl_df.iloc[AT_LARGE_COUNT+4:AT_LARGE_COUNT+8]["Team"].tolist()
n4o = atl_df.iloc[AT_LARGE_COUNT+8:AT_LARGE_COUNT+12]["Team"].tolist()

# Label buckets for visuals
df["Bubble_Bucket"] = ""
df.loc[df["Team"].isin(l4b), "Bubble_Bucket"] = "Last 4 Byes"
df.loc[df["Team"].isin(l4i), "Bubble_Bucket"] = "Last 4 In"
df.loc[df["Team"].isin(f4o), "Bubble_Bucket"] = "First 4 Out"
df.loc[df["Team"].isin(n4o), "Bubble_Bucket"] = "Next 4 Out"
df.loc[df["Is_AQ"], "Bubble_Bucket"] = "AQ"

# -----------------------------
# Visual outputs
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Field Size", FIELD_SIZE)
c2.metric("AQs", AQ_COUNT)
c3.metric("At-Larges", AT_LARGE_COUNT)
c4.metric("Bubble Cut", f"ATL #{AT_LARGE_COUNT}")

st.markdown("### Bubble Buckets")
b1, b2, b3, b4 = st.columns(4)
b1.write("**Last 4 Byes**"); b1.dataframe(df[df["Team"].isin(l4b)][["Team","Conference","Selection_Score","Seed_Score"]].sort_values("Selection_Score", ascending=False), use_container_width=True)
b2.write("**Last 4 In**");   b2.dataframe(df[df["Team"].isin(l4i)][["Team","Conference","Selection_Score","Seed_Score"]].sort_values("Selection_Score", ascending=False), use_container_width=True)
b3.write("**First 4 Out**"); b3.dataframe(df[df["Team"].isin(f4o)][["Team","Conference","Selection_Score","Seed_Score"]].sort_values("Selection_Score", ascending=False), use_container_width=True)
b4.write("**Next 4 Out**");  b4.dataframe(df[df["Team"].isin(n4o)][["Team","Conference","Selection_Score","Seed_Score"]].sort_values("Selection_Score", ascending=False), use_container_width=True)

st.markdown("### Full Projected Field (sorted by Seed Score)")
show_cols = ["Team","Conference","Is_AQ","Seed_Line","Selection_Score","Seed_Score","WIN_PCT","Bubble_Bucket"]
for extra in ["SOS SUM","SOS_SUM","Road Game","Neutral Site Game","Top 25 Opponent","March Madness Opponent","CLUTCH_SM","Coach Value","Sub 0 Clutch Performances"]:
    if extra in field_df.columns and extra not in show_cols:
        show_cols.append(extra)

st.dataframe(
    field_df.merge(df[["Team","Bubble_Bucket"]], on="Team", how="left")[show_cols]
        .sort_values(["Seed_Line","Seed_Score"], ascending=[True, False]),
    use_container_width=True
)

st.markdown("### At-Large Board (Top 80)")
st.dataframe(
    atl_df.head(80)[["ATL_Rank","Team","Conference","Selection_Score","Seed_Score","WIN_PCT"]],
    use_container_width=True
)

# Quick visual: conference counts in field
st.markdown("### Bids by Conference")
bids = df[df["In_Field"]].groupby("Conference", as_index=False)["Team"].count().rename(columns={"Team":"Bids"}).sort_values("Bids", ascending=False)
st.bar_chart(bids.set_index("Conference")["Bids"])
