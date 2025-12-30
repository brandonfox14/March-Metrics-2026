# APP/Pages/7_Bracketology.py
# =========================================================
# Bracketology — conference strength + AQ prediction + tier logic
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
DAILY_FILE     = os.path.join(BASE, "Daily_predictor_data-Table 1.csv")
COACH_FILE     = os.path.join(BASE, "Coach-Table 1.csv")

st.set_page_config(page_title="Bracketology", layout="wide")
st.title("Bracketology")

# -----------------------------
# Load helpers
# -----------------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def to_num(s: pd.Series, fill=np.nan) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fill)

def zscore(series: pd.Series) -> pd.Series:
    x = to_num(series, fill=np.nan)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(x)), index=series.index)
    return (x - mu) / sd

def within_group_pct_rank(df: pd.DataFrame, group_col: str, value_col: str, higher_is_better=True) -> pd.Series:
    """Percentile rank within each group: 0..1"""
    vals = to_num(df[value_col], fill=np.nan)
    asc = not higher_is_better
    return df.groupby(group_col)[value_col].transform(lambda s: to_num(s, fill=np.nan).rank(pct=True, ascending=asc).fillna(0.5))

def normalize_bool(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"true", "t", "1", "yes", "y"}

# -----------------------------
# Debug expander
# -----------------------------
with st.expander("Debug: paths / files", expanded=False):
    st.write("BASE exists:", os.path.exists(BASE))
    if os.path.exists(BASE):
        st.write("Files (first 80):", sorted(os.listdir(BASE))[:80])
    st.write("ALL_STATS_FILE exists:", os.path.exists(ALL_STATS_FILE))
    st.write("SOS_FILE exists:", os.path.exists(SOS_FILE))
    st.write("DAILY_FILE exists:", os.path.exists(DAILY_FILE))
    st.write("COACH_FILE exists:", os.path.exists(COACH_FILE))

# -----------------------------
# Load data
# -----------------------------
if not os.path.exists(ALL_STATS_FILE):
    st.error(f"Missing All Stats file: {ALL_STATS_FILE}")
    st.stop()
if not os.path.exists(SOS_FILE):
    st.error(f"Missing SOS file: {SOS_FILE}")
    st.stop()

all_stats = load_csv(ALL_STATS_FILE)
sos = load_csv(SOS_FILE)

daily = load_csv(DAILY_FILE) if os.path.exists(DAILY_FILE) else pd.DataFrame()
coach = load_csv(COACH_FILE) if os.path.exists(COACH_FILE) else pd.DataFrame()

# Standardize Team column
if "Teams" in all_stats.columns and "Team" not in all_stats.columns:
    all_stats = all_stats.rename(columns={"Teams": "Team"})
if "Teams" in sos.columns and "Team" not in sos.columns:
    sos = sos.rename(columns={"Teams": "Team"})
if not daily.empty and "Teams" in daily.columns and "Team" not in daily.columns:
    daily = daily.rename(columns={"Teams": "Team"})
if not coach.empty and "Teams" in coach.columns and "Team" not in coach.columns:
    coach = coach.rename(columns={"Teams": "Team"})

required = ["Team", "Conference", "Wins", "Losses", "Statistical Strength "]
missing = [c for c in required if c not in all_stats.columns]
if missing:
    st.error(f"All_Stats missing required columns: {missing}")
    st.stop()

# Win pct
all_stats["WIN_PCT"] = to_num(all_stats.get("WIN_PERC", all_stats["Wins"] / (all_stats["Wins"] + all_stats["Losses"])), fill=0)

# Historical Value (optional, but you asked to include)
if "Historical Value" not in all_stats.columns:
    all_stats["Historical Value"] = np.nan

# -----------------------------
# Conference strength (avg Statistical Strength)
# -----------------------------
conf_strength = (
    all_stats.groupby("Conference", as_index=False)["Statistical Strength "]
    .mean()
    .rename(columns={"Statistical Strength ": "Conf_StatStrength_Avg"})
)

# Higher conference strength = better
conf_strength["Conf_Rank"] = conf_strength["Conf_StatStrength_Avg"].rank(method="min", ascending=False).astype(int)
conf_strength = conf_strength.sort_values("Conf_Rank")

# Eligibility thresholds by conference rank
def conf_threshold(rank: int) -> float:
    if rank == 1: return 0.50
    if rank == 2: return 0.55
    if rank == 3: return 0.60
    if rank == 4: return 0.65
    if rank == 5: return 0.70
    return 0.75

conf_strength["AtLarge_WinPct_Threshold"] = conf_strength["Conf_Rank"].apply(conf_threshold)

# Merge conf strength into team table
df = all_stats.merge(conf_strength, on="Conference", how="left")

# -----------------------------
# SOS rollups (performance-based) -> pick a metric for tiering
# -----------------------------
# Aggregate SOS to team level.
if "Team" not in sos.columns:
    sos_team = pd.DataFrame({"Team": df["Team"].unique()})
else:
    sos_work = sos.copy()

    # Coerce likely numeric columns
    for col in ["SOS SUM","SOS_SUM","SOS Median","SOS_MED","SOS Min","SOS_MIN","SOS Max","SOS_MAX","SOS STDEV","SOS_STDEV"]:
        if col in sos_work.columns:
            sos_work[col] = to_num(sos_work[col], fill=np.nan)

    # If those appear row-wise, average them by team
    agg = {}
    for col in ["SOS SUM","SOS_SUM","SOS Median","SOS_MED","SOS Min","SOS_MIN","SOS Max","SOS_MAX","SOS STDEV","SOS_STDEV"]:
        if col in sos_work.columns:
            agg[col] = "mean"

    sos_team = sos_work.groupby("Team", as_index=False).agg(agg) if agg else pd.DataFrame({"Team": df["Team"].unique()})

df = df.merge(sos_team, on="Team", how="left")

# Choose one SOS metric for tiering (priority order)
SOS_TIER_COL = None
for cand in ["SOS SUM", "SOS_SUM", "SOS Median", "SOS_MED", "SOS Max", "SOS_MAX"]:
    if cand in df.columns:
        SOS_TIER_COL = cand
        break

if SOS_TIER_COL is None:
    df["SOS_TierValue"] = np.nan
else:
    df["SOS_TierValue"] = to_num(df[SOS_TIER_COL], fill=np.nan)

def tier_points(x: float) -> int:
    if np.isnan(x):
        return 0
    if x >= 100: return 4
    if x >= 50:  return 2
    if x >= 0:   return 1
    if x >= -50: return -1
    return -4

df["SOS_TierPoints"] = df["SOS_TierValue"].apply(tier_points)

# -----------------------------
# Within-conference team score (your stated logic)
# - tier points (absolute)
# - rank Statistical Strength (within conf)
# - rank WIN_PCT (within conf)
# - rank Historical Value (within conf) weighted at 0.5
# -----------------------------
df["StatStrength_rank_in_conf"] = within_group_pct_rank(df, "Conference", "Statistical Strength ", higher_is_better=True)
df["WinPct_rank_in_conf"]       = within_group_pct_rank(df, "Conference", "WIN_PCT", higher_is_better=True)
df["HistValue_rank_in_conf"]    = within_group_pct_rank(df, "Conference", "Historical Value", higher_is_better=True)

# Convert tier points to a 0..1-ish scale so it plays nicely with ranks
# (range -4..4) -> shift/scale to 0..1
df["Tier_scaled"] = (df["SOS_TierPoints"] + 4) / 8.0

# Team score (within conference)
df["Conf_TeamScore"] = (
    1.0 * df["Tier_scaled"] +
    1.0 * df["StatStrength_rank_in_conf"] +
    1.0 * df["WinPct_rank_in_conf"] +
    0.5 * df["HistValue_rank_in_conf"]
)

# -----------------------------
# AQ selection:
# 1) If coach table has AQ==TRUE for a team: lock it as AQ
# 2) Else: use Daily predictor to estimate "tournament winner power"
#    - build robust "PowerIndex" = average z-score across numeric columns
# -----------------------------
coach_aq = {}
if not coach.empty and "AQ" in coach.columns and "Conference" in coach.columns and "Team" in coach.columns:
    tmp = coach.copy()
    tmp["AQ_bool"] = tmp["AQ"].apply(normalize_bool)
    locked = tmp[tmp["AQ_bool"]].dropna(subset=["Conference","Team"])
    for _, r in locked.iterrows():
        coach_aq[r["Conference"]] = r["Team"]

daily_power = pd.DataFrame()
if not daily.empty:
    # merge conference if missing
    if "Conference" not in daily.columns:
        daily = daily.merge(df[["Team","Conference"]], on="Team", how="left")

    # build power index from numeric columns
    id_like = {"team","teams","conference","date","game","opponent","location"}
    numeric_cols = []
    for c in daily.columns:
        if c.lower() in id_like:
            continue
        if pd.api.types.is_numeric_dtype(daily[c]) or daily[c].dtype == object:
            # try numeric coercion
            s = pd.to_numeric(daily[c], errors="coerce")
            if s.notna().mean() >= 0.7:  # mostly numeric
                numeric_cols.append(c)

    if numeric_cols:
        zmat = pd.DataFrame({c: zscore(pd.to_numeric(daily[c], errors="coerce")) for c in numeric_cols})
        daily["PowerIndex"] = zmat.mean(axis=1).fillna(0)
        daily_power = (
            daily.groupby(["Conference","Team"], as_index=False)["PowerIndex"]
            .mean()
        )

# Pick AQs per conference
aq_map = {}
for conf in conf_strength["Conference"].tolist():
    # locked AQ?
    if conf in coach_aq:
        aq_map[conf] = coach_aq[conf]
        continue

    # use daily power if available
    if not daily_power.empty:
        pool = daily_power[daily_power["Conference"] == conf].copy()
        if not pool.empty:
            aq_map[conf] = pool.sort_values("PowerIndex", ascending=False)["Team"].iloc[0]
            continue

    # fallback: best Conf_TeamScore in conference
    pool2 = df[df["Conference"] == conf].copy()
    aq_map[conf] = pool2.sort_values("Conf_TeamScore", ascending=False)["Team"].iloc[0]

aqs = sorted(set(aq_map.values()))
df["Is_AQ"] = df["Team"].isin(aqs)

# -----------------------------
# At-large eligibility thresholds
# - determined by conference rank thresholds (50/55/60/65/70/75)
# -----------------------------
df["AtLarge_Eligible"] = df["WIN_PCT"] >= df["AtLarge_WinPct_Threshold"]

# -----------------------------
# Build field: 68 (AQ + at-larges)
# Better conference -> more bids naturally because:
#   - they have lower threshold
#   - and their teams have higher conf strength and score
# We'll select at-larges by an "Adjusted Score" that favors strong conferences.
# -----------------------------
FIELD_SIZE = 68
AQ_COUNT = len(aqs)
AT_LARGE_COUNT = FIELD_SIZE - AQ_COUNT

# Conference bonus (scaled 0..1): better conference => higher
conf_strength["ConfBonus"] = conf_strength["Conf_StatStrength_Avg"].rank(pct=True, ascending=True)
df = df.merge(conf_strength[["Conference","ConfBonus"]], on="Conference", how="left")

# Adjusted at-large score: team score + conf bonus
df["AtLarge_Score"] = df["Conf_TeamScore"] + 0.75 * df["ConfBonus"]

atl_pool = df[(~df["Is_AQ"]) & (df["AtLarge_Eligible"])].copy()
atl = (
    atl_pool.sort_values(["AtLarge_Score","Conf_TeamScore","WIN_PCT"], ascending=False)
            .head(AT_LARGE_COUNT)["Team"]
            .tolist()
)

field = set(aqs) | set(atl)
df["In_Field"] = df["Team"].isin(field)

# -----------------------------
# Seeding logic:
# "Top left 1 seed" = top team from best conference
# Next 1 seed slots come from the top of conference ladder / best remaining overall
# We'll create an overall Seed_Score that is consistent with your components.
# -----------------------------
# Seed score uses the same building blocks but lets conference strength help.
df["Seed_Score"] = (
    1.0 * df["Conf_TeamScore"] +
    0.50 * df["ConfBonus"]
)

field_df = df[df["In_Field"]].copy()

# Ensure #1 overall is top team from top conference (Conf_Rank = 1)
top_conf = conf_strength.sort_values("Conf_Rank")["Conference"].iloc[0]
top1_pool = field_df[field_df["Conference"] == top_conf].copy()
if not top1_pool.empty:
    top1_team = top1_pool.sort_values("Seed_Score", ascending=False)["Team"].iloc[0]
else:
    top1_team = field_df.sort_values("Seed_Score", ascending=False)["Team"].iloc[0]

# Rank teams, forcing #1 overall to be that top conference top team
field_df = field_df.sort_values("Seed_Score", ascending=False).reset_index(drop=True)
if top1_team in field_df["Team"].values:
    # move it to top
    top_row = field_df[field_df["Team"] == top1_team]
    rest = field_df[field_df["Team"] != top1_team]
    field_df = pd.concat([top_row, rest], ignore_index=True)

field_df["Overall_Rank"] = np.arange(1, len(field_df) + 1)
field_df["Seed_Line"] = (((field_df["Overall_Rank"] - 1) // 4) + 1).clip(1, 16)

# -----------------------------
# Bubble buckets (based on at-large board)
# -----------------------------
atl_board = df[~df["Is_AQ"]].copy()
# Keep ineligible teams, but they will naturally fall down (still useful to see)
atl_board = atl_board.sort_values(["AtLarge_Score","Conf_TeamScore","WIN_PCT"], ascending=False).reset_index(drop=True)
atl_board["ATL_Rank"] = np.arange(1, len(atl_board) + 1)

l4b = atl_board.iloc[max(AT_LARGE_COUNT-4, 0):AT_LARGE_COUNT]["Team"].tolist()
l4i = atl_board.iloc[AT_LARGE_COUNT:AT_LARGE_COUNT+4]["Team"].tolist()
f4o = atl_board.iloc[AT_LARGE_COUNT+4:AT_LARGE_COUNT+8]["Team"].tolist()
n4o = atl_board.iloc[AT_LARGE_COUNT+8:AT_LARGE_COUNT+12]["Team"].tolist()

df["Bubble_Bucket"] = ""
df.loc[df["Team"].isin(l4b), "Bubble_Bucket"] = "Last 4 Byes"
df.loc[df["Team"].isin(l4i), "Bubble_Bucket"] = "Last 4 In"
df.loc[df["Team"].isin(f4o), "Bubble_Bucket"] = "First 4 Out"
df.loc[df["Team"].isin(n4o), "Bubble_Bucket"] = "Next 4 Out"
df.loc[df["Is_AQ"], "Bubble_Bucket"] = "AQ"

# -----------------------------
# VISUALS
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Field Size", FIELD_SIZE)
c2.metric("AQs", AQ_COUNT)
c3.metric("At-Larges", AT_LARGE_COUNT)
c4.metric("Top Conference", f"{top_conf}")

st.markdown("## Conference Strength (by avg Statistical Strength)")
st.dataframe(
    conf_strength[["Conference","Conf_StatStrength_Avg","Conf_Rank","AtLarge_WinPct_Threshold"]]
    .sort_values("Conf_Rank"),
    use_container_width=True,
    height=400
)

st.markdown("## Automatic Qualifiers (AQ)")
aq_df = pd.DataFrame({"Conference": list(aq_map.keys()), "AQ Team": list(aq_map.values())})
aq_df = aq_df.merge(conf_strength[["Conference","Conf_Rank"]], on="Conference", how="left").sort_values("Conf_Rank")
st.dataframe(aq_df, use_container_width=True, height=450)

st.markdown("## Bubble Buckets")
b1, b2, b3, b4 = st.columns(4)

def bucket_table(title, teams):
    sub = df[df["Team"].isin(teams)][[
        "Team","Conference","WIN_PCT","AtLarge_Eligible","AtLarge_WinPct_Threshold",
        "SOS_TierValue","SOS_TierPoints","Statistical Strength ","Historical Value",
        "Conf_TeamScore","AtLarge_Score"
    ]].sort_values("AtLarge_Score", ascending=False)
    st.write(f"**{title}**")
    st.dataframe(sub, use_container_width=True, height=280)

with b1: bucket_table("Last 4 Byes", l4b)
with b2: bucket_table("Last 4 In", l4i)
with b3: bucket_table("First 4 Out", f4o)
with b4: bucket_table("Next 4 Out", n4o)

st.markdown("## Full Projected Field (Seeded)")
show_cols = [
    "Overall_Rank","Seed_Line","Team","Conference","Is_AQ",
    "WIN_PCT","AtLarge_Eligible","AtLarge_WinPct_Threshold",
    "SOS_TierValue","SOS_TierPoints",
    "Statistical Strength ","Historical Value",
    "Conf_TeamScore","AtLarge_Score","Seed_Score","Bubble_Bucket"
]
if SOS_TIER_COL and SOS_TIER_COL not in show_cols:
    pass  # already represented via SOS_TierValue

out = field_df.merge(df[["Team","Bubble_Bucket","AtLarge_Eligible","AtLarge_WinPct_Threshold","AtLarge_Score"]], on="Team", how="left")
st.dataframe(out[show_cols].sort_values(["Seed_Line","Seed_Score"], ascending=[True, False]), use_container_width=True, height=750)

st.markdown("## At-Large Board (Top 120)")
st.dataframe(
    atl_board.head(120)[[
        "ATL_Rank","Team","Conference","WIN_PCT","AtLarge_Eligible","AtLarge_WinPct_Threshold",
        "SOS_TierValue","SOS_TierPoints","Statistical Strength ","Historical Value",
        "Conf_TeamScore","AtLarge_Score","Bubble_Bucket"
    ]],
    use_container_width=True,
    height=750
)

st.markdown("## Bids by Conference")
bids = df[df["In_Field"]].groupby("Conference", as_index=False)["Team"].count().rename(columns={"Team":"Bids"})
bids = bids.merge(conf_strength[["Conference","Conf_Rank"]], on="Conference", how="left").sort_values(["Bids","Conf_Rank"], ascending=[False, True])
st.bar_chart(bids.set_index("Conference")["Bids"])
