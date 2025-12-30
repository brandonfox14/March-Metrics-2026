# APP/Pages/7_Bracketology.py
# =========================================================
# Bracketology — AQs + At-larges + Bubble Buckets
# =========================================================
import os
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Paths
# -----------------------------
BASE = "Data/26_March_Madness_Databook"
ALL_STATS_FILE = os.path.join(BASE, "All_Stats-THE_TABLE.csv")
SOS_FILE       = os.path.join(BASE, "SOS-Table 1.csv")

st.set_page_config(page_title="Bracketology", layout="wide")
st.title("Bracketology")

# -----------------------------
# Debug (helps on Streamlit Cloud)
# -----------------------------
with st.expander("Debug: paths / files", expanded=False):
    st.write("BASE exists:", os.path.exists(BASE))
    if os.path.exists(BASE):
        st.write("Files (first 60):", sorted(os.listdir(BASE))[:60])
    st.write("ALL_STATS_FILE exists:", os.path.exists(ALL_STATS_FILE))
    st.write("SOS_FILE exists:", os.path.exists(SOS_FILE))

# -----------------------------
# Load data
# -----------------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

if not os.path.exists(ALL_STATS_FILE):
    st.error(f"Missing All Stats file: {ALL_STATS_FILE}")
    st.stop()
if not os.path.exists(SOS_FILE):
    st.error(f"Missing SOS file: {SOS_FILE}")
    st.stop()

all_stats = load_csv(ALL_STATS_FILE)
sos = load_csv(SOS_FILE)

# Standardize team column name
if "Teams" in all_stats.columns and "Team" not in all_stats.columns:
    all_stats = all_stats.rename(columns={"Teams": "Team"})
if "Teams" in sos.columns and "Team" not in sos.columns:
    sos = sos.rename(columns={"Teams": "Team"})

required_all = ["Team", "Conference", "Wins", "Losses"]
missing = [c for c in required_all if c not in all_stats.columns]
if missing:
    st.error(f"All_Stats is missing required columns: {missing}")
    st.stop()

# -----------------------------
# Helpers
# -----------------------------
def to_num(s: pd.Series, fill=0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fill)

def pct_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    s = to_num(series, fill=np.nan)
    r = s.rank(pct=True, ascending=not higher_is_better)
    # fill NaNs to middle so missing cols don't dominate
    return r.fillna(0.5)

def rank_to_quality(rank_series: pd.Series) -> pd.Series:
    # lower numeric rank => better
    r = to_num(rank_series, fill=np.nan)
    return (-r).rank(pct=True).fillna(0.5)

# -----------------------------
# SOS -> Team rollups (résumé)
# -----------------------------
if "Team" not in sos.columns:
    st.warning("SOS is missing Team column — résumé rollups disabled (Selection will be mostly All_Stats based).")
    sos_team = pd.DataFrame({"Team": all_stats["Team"].unique()})
else:
    sos_work = sos.copy()

    # numeric columns we may aggregate if present
    sum_cols = [
        "Road Game", "Neutral Site Game", "Top 25 Opponent", "March Madness Opponent",
        "Tournament Caliber Game", "Won as Underdog", "Lost as Favorite",
        "Win Game within 8", "Within 10 line", "Line over 20", "Cover Spread"
    ]
    mean_cols = [
        "(SM+Line)^2", "(SM+Line+10)^2", "Volatility",
        "SOS Median", "SOS Min", "SOS Max", "SOS STDEV", "SOS SUM",
        "SOS_MED", "SOS_MIN", "SOS_MAX", "SOS_STDEV", "SOS_SUM"
    ]

    for col in set(sum_cols + mean_cols):
        if col in sos_work.columns:
            sos_work[col] = to_num(sos_work[col], fill=0)

    agg = {}
    for col in sum_cols:
        if col in sos_work.columns:
            agg[col] = "sum"
    for col in mean_cols:
        if col in sos_work.columns:
            agg[col] = "mean"

    sos_team = sos_work.groupby("Team", as_index=False).agg(agg) if agg else pd.DataFrame({"Team": all_stats["Team"].unique()})

# Merge
df = all_stats.merge(sos_team, on="Team", how="left")

# -----------------------------
# Compute scores
# -----------------------------
# Win pct
df["WIN_PCT"] = to_num(df.get("WIN_PERC", df["Wins"] / (df["Wins"] + df["Losses"])), fill=0)

# ---- Selection Score (IN/OUT)
selection_components = {}

# SOS sum (either style)
if "SOS SUM" in df.columns:
    selection_components["sos_sum"] = pct_rank(df["SOS SUM"], True)
elif "SOS_SUM" in df.columns:
    selection_components["sos_sum"] = pct_rank(df["SOS_SUM"], True)

# résumé volume/opponent quality
for col, key, hib in [
    ("Road Game", "road_games", True),
    ("Neutral Site Game", "neutral_games", True),
    ("Top 25 Opponent", "top25_opp", True),
    ("March Madness Opponent", "mm_opp", True),
    ("Tournament Caliber Game", "tcg", True),
    ("Won as Underdog", "won_dog", True),
]:
    if col in df.columns:
        selection_components[key] = pct_rank(df[col], hib)

# downside
if "Lost as Favorite" in df.columns:
    selection_components["lost_fav"] = pct_rank(df["Lost as Favorite"], False)

# your edge: clutch + coach
if "CLUTCH_SM" in df.columns:
    selection_components["clutch_sm"] = pct_rank(df["CLUTCH_SM"], True)
if "Sub 0 Clutch Performances" in df.columns:
    selection_components["sub0_clutch"] = pct_rank(df["Sub 0 Clutch Performances"], False)
if "Coach Value" in df.columns:
    selection_components["coach_value"] = pct_rank(df["Coach Value"], True)

selection_components["win_pct"] = pct_rank(df["WIN_PCT"], True)

# weights (tune as you like)
w_sel = {
    "sos_sum": 0.20,
    "road_games": 0.07,
    "neutral_games": 0.03,
    "top25_opp": 0.06,
    "mm_opp": 0.06,
    "tcg": 0.06,
    "won_dog": 0.06,
    "lost_fav": 0.08,
    "clutch_sm": 0.14,
    "sub0_clutch": 0.12,
    "coach_value": 0.06,
    "win_pct": 0.12,
}

sel_score = np.zeros(len(df))
w_used = 0.0
for k, w in w_sel.items():
    if k in selection_components:
        sel_score += w * selection_components[k].to_numpy()
        w_used += w

df["Selection_Score"] = 100 * (sel_score / max(w_used, 1e-9))

# ---- Seed Score (quality)
seed_components = {}

# Use ranks if available
if "Off_eff_hybrid_rank" in df.columns:
    seed_components["off_rank"] = rank_to_quality(df["Off_eff_hybrid_rank"])
elif "Off_eff_rank" in df.columns:
    seed_components["off_rank"] = rank_to_quality(df["Off_eff_rank"])

if "Def_eff_hybrid_rank" in df.columns:
    seed_components["def_rank"] = rank_to_quality(df["Def_eff_hybrid_rank"])

if "SM" in df.columns:
    seed_components["sm"] = pct_rank(df["SM"], True)

if "Rebound Rate" in df.columns:
    seed_components["reb_rate"] = pct_rank(df["Rebound Rate"], True)
elif "Rebound Rate Rank" in df.columns:
    seed_components["reb_rate"] = rank_to_quality(df["Rebound Rate Rank"])

if "TO Rank" in df.columns:
    seed_components["to_rank"] = rank_to_quality(df["TO Rank"])

if "Volatility" in df.columns:
    seed_components["volatility"] = pct_rank(df["Volatility"], False)

w_seed = {
    "off_rank": 0.24,
    "def_rank": 0.24,
    "sm": 0.20,
    "reb_rate": 0.10,
    "to_rank": 0.10,
    "volatility": 0.12,
}

seed_score = np.zeros(len(df))
w2 = 0.0
for k, w in w_seed.items():
    if k in seed_components:
        seed_score += w * seed_components[k].to_numpy()
        w2 += w

df["Seed_Score"] = 100 * (seed_score / max(w2, 1e-9))

# -----------------------------
# AQ selection UI (manual for now)
# -----------------------------
st.subheader("Automatic Qualifiers (pick one team per conference)")

conf_list = sorted(df["Conference"].dropna().unique().tolist())
aq_map = {}
cols = st.columns(4)

for i, conf in enumerate(conf_list):
    c = cols[i % 4]
    teams = df.loc[df["Conference"] == conf, "Team"].sort_values().tolist()
    if not teams:
        continue
    # default to best Selection Score in conference
    best_team = df.loc[df["Conference"] == conf].sort_values("Selection_Score", ascending=False)["Team"].iloc[0]
    default_idx = teams.index(best_team) if best_team in teams else 0
    pick = c.selectbox(conf, teams, index=default_idx, key=f"aq_{conf}")
    aq_map[conf] = pick

aqs = sorted(set(aq_map.values()))
df["Is_AQ"] = df["Team"].isin(aqs)

# -----------------------------
# Build Field
# -----------------------------
FIELD_SIZE = 68
AQ_COUNT = len(aqs)
AT_LARGE_COUNT = FIELD_SIZE - AQ_COUNT

atl = (
    df.loc[~df["Is_AQ"]]
      .sort_values(["Selection_Score", "Seed_Score"], ascending=False)
      .head(AT_LARGE_COUNT)["Team"]
      .tolist()
)

field = set(aqs) | set(atl)
df["In_Field"] = df["Team"].isin(field)

# S-curve order
field_df = df[df["In_Field"]].copy().sort_values("Seed_Score", ascending=False)
field_df["Overall_Rank"] = np.arange(1, len(field_df) + 1)
field_df["Seed_Line"] = (((field_df["Overall_Rank"] - 1) // 4) + 1).clip(1, 16)

# At-large board and bubble
atl_df = df.loc[~df["Is_AQ"]].sort_values(["Selection_Score", "Seed_Score"], ascending=False).copy()
atl_df["ATL_Rank"] = np.arange(1, len(atl_df) + 1)

l4b = atl_df.iloc[max(AT_LARGE_COUNT-4, 0):AT_LARGE_COUNT]["Team"].tolist()
l4i = atl_df.iloc[AT_LARGE_COUNT:AT_LARGE_COUNT+4]["Team"].tolist()
f4o = atl_df.iloc[AT_LARGE_COUNT+4:AT_LARGE_COUNT+8]["Team"].tolist()
n4o = atl_df.iloc[AT_LARGE_COUNT+8:AT_LARGE_COUNT+12]["Team"].tolist()

df["Bubble_Bucket"] = ""
df.loc[df["Team"].isin(l4b), "Bubble_Bucket"] = "Last 4 Byes"
df.loc[df["Team"].isin(l4i), "Bubble_Bucket"] = "Last 4 In"
df.loc[df["Team"].isin(f4o), "Bubble_Bucket"] = "First 4 Out"
df.loc[df["Team"].isin(n4o), "Bubble_Bucket"] = "Next 4 Out"
df.loc[df["Is_AQ"], "Bubble_Bucket"] = "AQ"

# -----------------------------
# Visuals
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Field Size", FIELD_SIZE)
c2.metric("AQs", AQ_COUNT)
c3.metric("At-Larges", AT_LARGE_COUNT)
c4.metric("Bubble Cut", f"ATL #{AT_LARGE_COUNT}")

st.markdown("### Bubble Buckets")
b1, b2, b3, b4 = st.columns(4)

def show_bucket(title, teams):
    sub = df[df["Team"].isin(teams)][["Team","Conference","Selection_Score","Seed_Score","WIN_PCT"]].sort_values("Selection_Score", ascending=False)
    st.write(f"**{title}**")
    st.dataframe(sub, use_container_width=True, height=220)

with b1: show_bucket("Last 4 Byes", l4b)
with b2: show_bucket("Last 4 In", l4i)
with b3: show_bucket("First 4 Out", f4o)
with b4: show_bucket("Next 4 Out", n4o)

st.markdown("### Full Projected Field (sorted by Seed Line / Seed Score)")

show_cols = ["Team","Conference","Is_AQ","Seed_Line","Selection_Score","Seed_Score","WIN_PCT","Bubble_Bucket"]
for extra in ["SOS_SUM","SOS SUM","Road Game","Neutral Site Game","Top 25 Opponent","March Madness Opponent",
              "CLUTCH_SM","Coach Value","Sub 0 Clutch Performances","Off_eff_hybrid_rank","Def_eff_hybrid_rank","SM"]:
    if extra in field_df.columns and extra not in show_cols:
        show_cols.append(extra)

out = field_df.merge(df[["Team","Bubble_Bucket"]], on="Team", how="left")
st.dataframe(out[show_cols].sort_values(["Seed_Line","Seed_Score"], ascending=[True, False]), use_container_width=True, height=650)

st.markdown("### At-Large Board (Top 80)")
st.dataframe(atl_df.head(80)[["ATL_Rank","Team","Conference","Selection_Score","Seed_Score","WIN_PCT","Bubble_Bucket"]], use_container_width=True, height=650)

st.markdown("### Bids by Conference")
bids = df[df["In_Field"]].groupby("Conference", as_index=False)["Team"].count().rename(columns={"Team":"Bids"}).sort_values("Bids", ascending=False)
st.bar_chart(bids.set_index("Conference")["Bids"])
