# APP/Pages/7_Bracketology.py
# =========================================================
# Bracketology — Conference Strength + AQ Prediction + Bubble + First Four
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
# Utilities
# -----------------------------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def to_num(s: pd.Series, fill=np.nan) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(fill)

def zscore(series: pd.Series) -> pd.Series:
    x = to_num(series, fill=np.nan)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(x)), index=series.index)
    return (x - mu) / sd

def within_group_pct_rank(df: pd.DataFrame, group_col: str, value_col: str, higher_is_better: bool = True) -> pd.Series:
    """
    Percentile rank within each group, scaled 0..1.
    higher_is_better=True => larger values get higher rank.
    higher_is_better=False => smaller values get higher rank.
    """
    asc = not higher_is_better
    return df.groupby(group_col)[value_col].transform(
        lambda s: to_num(s, fill=np.nan).rank(pct=True, ascending=asc).fillna(0.5)
    )

def normalize_bool(x) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"true", "t", "1", "yes", "y"}

def pick_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None

# -----------------------------
# Debug: paths
# -----------------------------
with st.expander("Debug: paths / files", expanded=False):
    st.write("BASE exists:", os.path.exists(BASE))
    if os.path.exists(BASE):
        st.write("Files (first 80):", sorted(os.listdir(BASE))[:80])
    st.write("ALL_STATS_FILE:", ALL_STATS_FILE, "exists:", os.path.exists(ALL_STATS_FILE))
    st.write("SOS_FILE:", SOS_FILE, "exists:", os.path.exists(SOS_FILE))
    st.write("DAILY_FILE:", DAILY_FILE, "exists:", os.path.exists(DAILY_FILE))
    st.write("COACH_FILE:", COACH_FILE, "exists:", os.path.exists(COACH_FILE))

# -----------------------------
# Load data
# -----------------------------
if not os.path.exists(ALL_STATS_FILE):
    st.error(f"Missing file: {ALL_STATS_FILE}")
    st.stop()
if not os.path.exists(SOS_FILE):
    st.error(f"Missing file: {SOS_FILE}")
    st.stop()

all_stats = clean_cols(load_csv(ALL_STATS_FILE))
sos = clean_cols(load_csv(SOS_FILE))

daily = clean_cols(load_csv(DAILY_FILE)) if os.path.exists(DAILY_FILE) else pd.DataFrame()
coach = clean_cols(load_csv(COACH_FILE)) if os.path.exists(COACH_FILE) else pd.DataFrame()

# Standardize Team column naming
if "Teams" in all_stats.columns and "Team" not in all_stats.columns:
    all_stats = all_stats.rename(columns={"Teams": "Team"})
if "Teams" in sos.columns and "Team" not in sos.columns:
    sos = sos.rename(columns={"Teams": "Team"})
if not daily.empty and "Teams" in daily.columns and "Team" not in daily.columns:
    daily = daily.rename(columns={"Teams": "Team"})
if not coach.empty and "Teams" in coach.columns and "Team" not in coach.columns:
    coach = coach.rename(columns={"Teams": "Team"})

# Resolve key All_Stats columns (handles small naming drift)
COL_TEAM = "Team"
COL_CONF = pick_existing(all_stats, ["Conference", "Conf"])
COL_WINS = pick_existing(all_stats, ["Wins", "W"])
COL_LOSSES = pick_existing(all_stats, ["Losses", "L"])
COL_WINPERC = pick_existing(all_stats, ["WIN_PERC", "WIN_PCT", "Win %", "Win Perc"])
COL_STAT_STRENGTH = pick_existing(all_stats, ["Statistical Strength", "Statistical Strength ", "Stat_Strength", "StatStrength"])
COL_HIST_VALUE = pick_existing(all_stats, ["Historical Value", "Historical Value ", "History Value"])
COL_CHAMP_CRIT = pick_existing(all_stats, ["Championship Criteria", "Championship Criteria ", "Champ Criteria"])

required_missing = []
if COL_TEAM not in all_stats.columns:
    required_missing.append("Team/Teams")
if COL_CONF is None:
    required_missing.append("Conference")
if COL_WINS is None:
    required_missing.append("Wins")
if COL_LOSSES is None:
    required_missing.append("Losses")
if COL_STAT_STRENGTH is None:
    required_missing.append("Statistical Strength")
if required_missing:
    st.error(f"All_Stats missing required columns: {required_missing}")
    st.stop()

# Ensure optional columns exist
if COL_HIST_VALUE is None:
    all_stats["Historical Value"] = np.nan
    COL_HIST_VALUE = "Historical Value"
if COL_CHAMP_CRIT is None:
    all_stats["Championship Criteria"] = np.nan
    COL_CHAMP_CRIT = "Championship Criteria"

# Win pct
if COL_WINPERC is not None:
    all_stats["WIN_PCT"] = to_num(all_stats[COL_WINPERC], fill=0)
else:
    w = to_num(all_stats[COL_WINS], fill=0)
    l = to_num(all_stats[COL_LOSSES], fill=0)
    all_stats["WIN_PCT"] = (w / (w + l)).replace([np.inf, -np.inf], 0).fillna(0)

# Rename for internal consistency
all_stats = all_stats.rename(columns={
    COL_CONF: "Conference",
    COL_WINS: "Wins",
    COL_LOSSES: "Losses",
    COL_STAT_STRENGTH: "Statistical_Strength",
    COL_HIST_VALUE: "Historical_Value",
    COL_CHAMP_CRIT: "Championship_Criteria",
})

# -----------------------------
# Conference Strength
# - Average Statistical_Strength (but lower is better for teams; conference "strength" should be better when avg is lower)
# We'll invert when ranking conferences.
# -----------------------------
conf_strength = (
    all_stats.groupby("Conference", as_index=False)["Statistical_Strength"]
    .mean()
    .rename(columns={"Statistical_Strength": "Conf_StatStrength_Avg"})
)

# Conference rank: lower avg Statistical Strength = stronger conference
conf_strength["Conf_Rank"] = conf_strength["Conf_StatStrength_Avg"].rank(method="min", ascending=True).astype(int)
conf_strength = conf_strength.sort_values("Conf_Rank")

def conf_threshold(rank: int) -> float:
    if rank == 1: return 0.50
    if rank == 2: return 0.55
    if rank == 3: return 0.60
    if rank == 4: return 0.65
    if rank == 5: return 0.70
    return 0.75

conf_strength["AtLarge_WinPct_Threshold"] = conf_strength["Conf_Rank"].apply(conf_threshold)

df = all_stats.merge(conf_strength, on="Conference", how="left")

# Conference bonus: better conference -> higher bonus
# Since Conf_Rank lower is better, invert percentile
conf_strength["ConfBonus"] = (-conf_strength["Conf_StatStrength_Avg"]).rank(pct=True, ascending=True)
df = df.merge(conf_strength[["Conference", "ConfBonus"]], on="Conference", how="left")

# -----------------------------
# SOS rollup + tiers
# SOS is "performance based" — we'll tier a single selected SOS metric.
# Priority: SOS SUM/SOS_SUM, else SOS Median/SOS_MED, else SOS Max/SOS_MAX
# -----------------------------
if "Teams" in sos.columns and "Team" not in sos.columns:
    sos = sos.rename(columns={"Teams": "Team"})

SOS_TIER_COL = pick_existing(sos, ["SOS SUM", "SOS_SUM", "SOS Median", "SOS_MED", "SOS Max", "SOS_MAX"])
if SOS_TIER_COL is None:
    # still allow app to run, tiers become 0
    sos_team = pd.DataFrame({"Team": df["Team"].unique()})
    df["SOS_TierValue"] = np.nan
else:
    sos_work = sos.copy()
    sos_work[SOS_TIER_COL] = to_num(sos_work[SOS_TIER_COL], fill=np.nan)

    sos_team = sos_work.groupby("Team", as_index=False)[SOS_TIER_COL].mean().rename(columns={SOS_TIER_COL: "SOS_TierValue"})
    df = df.merge(sos_team, on="Team", how="left")

def tier_points(x: float) -> int:
    if pd.isna(x):
        return 0
    if x >= 100: return 4
    if x >= 50:  return 2
    if x >= 0:   return 1
    if x >= -50: return -1
    return -4

df["SOS_TierPoints"] = df["SOS_TierValue"].apply(tier_points)

# -----------------------------
# Within-conference ranking formula (ALL vs conference opponents)
# Directions you specified:
# - TierPoints: higher = better
# - WIN_PCT: higher = better
# - Statistical_Strength: lower = better
# - Historical_Value: higher = better (half weight)
# - Championship_Criteria: included at quarter weight
# -----------------------------
df["Tier_rank_in_conf"] = within_group_pct_rank(df, "Conference", "SOS_TierPoints", higher_is_better=True)
df["WinPct_rank_in_conf"] = within_group_pct_rank(df, "Conference", "WIN_PCT", higher_is_better=True)
df["StatStrength_rank_in_conf"] = within_group_pct_rank(df, "Conference", "Statistical_Strength", higher_is_better=False)
df["HistValue_rank_in_conf"] = within_group_pct_rank(df, "Conference", "Historical_Value", higher_is_better=True)
df["ChampCrit_rank_in_conf"] = within_group_pct_rank(df, "Conference", "Championship_Criteria", higher_is_better=True)

df["Conf_TeamScore"] = (
    1.00 * df["Tier_rank_in_conf"] +
    1.00 * df["WinPct_rank_in_conf"] +
    1.00 * df["StatStrength_rank_in_conf"] +
    0.50 * df["HistValue_rank_in_conf"] +
    0.25 * df["ChampCrit_rank_in_conf"]
)

# -----------------------------
# AQ Selection
# - If Coach file has AQ==TRUE => lock that team as AQ
# - Else use Daily predictor to estimate who wins conference tourney (power index)
#   (generic, schema-proof): average z-score across numeric-ish columns
# - Else fallback: top Conf_TeamScore in conference
# -----------------------------
coach_aq = {}
if not coach.empty and {"AQ", "Team"}.issubset(set(coach.columns)):
    # If coach doesn't carry Conference, merge it in from df
    if "Conference" not in coach.columns:
        coach = coach.merge(df[["Team", "Conference"]], on="Team", how="left")
    if "Conference" in coach.columns:
        tmp = coach.copy()
        tmp["AQ_bool"] = tmp["AQ"].apply(normalize_bool)
        locked = tmp[tmp["AQ_bool"]].dropna(subset=["Conference", "Team"])
        for _, r in locked.iterrows():
            coach_aq[r["Conference"]] = r["Team"]

daily_power = pd.DataFrame()
if not daily.empty and "Team" in daily.columns:
    if "Conference" not in daily.columns:
        daily = daily.merge(df[["Team", "Conference"]], on="Team", how="left")

    # numeric-ish columns (skip obvious ids)
    id_like = {"team", "teams", "conference", "date", "game", "opponent", "location", "home", "away"}
    numeric_cols = []
    for c in daily.columns:
        if c.lower() in id_like:
            continue
        s = pd.to_numeric(daily[c], errors="coerce")
        if s.notna().mean() >= 0.70:  # mostly numeric
            numeric_cols.append(c)

    if numeric_cols:
        zmat = pd.DataFrame({c: zscore(pd.to_numeric(daily[c], errors="coerce")) for c in numeric_cols})
        daily["PowerIndex"] = zmat.mean(axis=1).fillna(0)
        daily_power = daily.groupby(["Conference", "Team"], as_index=False)["PowerIndex"].mean()

aq_map = {}
for conf in conf_strength["Conference"].tolist():
    # 1) locked AQ
    if conf in coach_aq:
        aq_map[conf] = coach_aq[conf]
        continue
    # 2) predicted AQ from daily
    if not daily_power.empty:
        pool = daily_power[daily_power["Conference"] == conf].copy()
        if not pool.empty:
            aq_map[conf] = pool.sort_values("PowerIndex", ascending=False)["Team"].iloc[0]
            continue
    # 3) fallback
    pool2 = df[df["Conference"] == conf].copy()
    aq_map[conf] = pool2.sort_values("Conf_TeamScore", ascending=False)["Team"].iloc[0]

aqs = sorted(set(aq_map.values()))
df["Is_AQ"] = df["Team"].isin(aqs)

# -----------------------------
# At-large eligibility thresholds (by conference rank)
# -----------------------------
df["AtLarge_Eligible"] = df["WIN_PCT"] >= df["AtLarge_WinPct_Threshold"]

# -----------------------------
# Build Field (68)
# - AQs always in
# - At-larges chosen from eligible non-AQs by AtLarge_Score
#   AtLarge_Score uses Conf_TeamScore + conference bonus
# -----------------------------
FIELD_SIZE = 68
AQ_COUNT = len(aqs)
AT_LARGE_COUNT = FIELD_SIZE - AQ_COUNT

df["AtLarge_Score"] = df["Conf_TeamScore"] + 0.75 * df["ConfBonus"]

atl_pool = df[(~df["Is_AQ"]) & (df["AtLarge_Eligible"])].copy()
atl = (
    atl_pool.sort_values(["AtLarge_Score", "Conf_TeamScore", "WIN_PCT"], ascending=False)
            .head(AT_LARGE_COUNT)["Team"]
            .tolist()
)

field = set(aqs) | set(atl)
df["In_Field"] = df["Team"].isin(field)

# -----------------------------
# Seeding
# - Overall seed score: Conf_TeamScore + 0.5*ConfBonus (tune if you want)
# - Force #1 overall = top team from top conference (Conf_Rank=1)
# -----------------------------
df["Seed_Score"] = df["Conf_TeamScore"] + 0.50 * df["ConfBonus"]

field_df = df[df["In_Field"]].copy().sort_values("Seed_Score", ascending=False).reset_index(drop=True)

top_conf = conf_strength.sort_values("Conf_Rank")["Conference"].iloc[0]
top1_pool = field_df[field_df["Conference"] == top_conf].copy()
if not top1_pool.empty:
    top1_team = top1_pool.sort_values("Seed_Score", ascending=False)["Team"].iloc[0]
else:
    top1_team = field_df.sort_values("Seed_Score", ascending=False)["Team"].iloc[0]

if top1_team in field_df["Team"].values:
    top_row = field_df[field_df["Team"] == top1_team]
    rest = field_df[field_df["Team"] != top1_team]
    field_df = pd.concat([top_row, rest], ignore_index=True)

field_df["Overall_Rank"] = np.arange(1, len(field_df) + 1)
field_df["Seed_Line"] = (((field_df["Overall_Rank"] - 1) // 4) + 1).clip(1, 16)

# -----------------------------
# Bubble buckets (based on at-large board ordering)
# -----------------------------
atl_board = df[~df["Is_AQ"]].copy()
atl_board = atl_board.sort_values(["AtLarge_Score", "Conf_TeamScore", "WIN_PCT"], ascending=False).reset_index(drop=True)
atl_board["ATL_Rank"] = np.arange(1, len(atl_board) + 1)

l4b = atl_board.iloc[max(AT_LARGE_COUNT - 4, 0):AT_LARGE_COUNT]["Team"].tolist()
l4i = atl_board.iloc[AT_LARGE_COUNT:AT_LARGE_COUNT + 4]["Team"].tolist()
f4o = atl_board.iloc[AT_LARGE_COUNT + 4:AT_LARGE_COUNT + 8]["Team"].tolist()
n4o = atl_board.iloc[AT_LARGE_COUNT + 8:AT_LARGE_COUNT + 12]["Team"].tolist()

df["Bubble_Bucket"] = ""
df.loc[df["Team"].isin(l4b), "Bubble_Bucket"] = "Last 4 Byes"
df.loc[df["Team"].isin(l4i), "Bubble_Bucket"] = "Last 4 In"
df.loc[df["Team"].isin(f4o), "Bubble_Bucket"] = "First 4 Out"
df.loc[df["Team"].isin(n4o), "Bubble_Bucket"] = "Next 4 Out"
df.loc[df["Is_AQ"], "Bubble_Bucket"] = "AQ"

# -----------------------------
# First Four logic
# - 4 lowest-seeded AQs play each other (two games) for the 16 seed slots
# - 4 lowest-seeded At-Larges in the field play each other (two games)
# -----------------------------
aq_field = field_df[field_df["Is_AQ"]].copy()
al_field = field_df[~field_df["Is_AQ"]].copy()

lowest_4_aq = aq_field.sort_values(["Seed_Score", "Overall_Rank"], ascending=[True, False]).head(4).copy()
lowest_4_al = al_field.sort_values(["Seed_Score", "Overall_Rank"], ascending=[True, False]).head(4).copy()

field_df["FirstFour_Type"] = ""
field_df.loc[field_df["Team"].isin(lowest_4_aq["Team"]), "FirstFour_Type"] = "First Four (AQ)"
field_df.loc[field_df["Team"].isin(lowest_4_al["Team"]), "FirstFour_Type"] = "First Four (At-Large)"

def make_matchups(subdf: pd.DataFrame, label: str) -> pd.DataFrame:
    subdf = subdf.sort_values(["Seed_Score", "Overall_Rank"], ascending=[True, False]).reset_index(drop=True)
    if len(subdf) < 4:
        return pd.DataFrame(columns=["Game", "Type", "Team A", "Team B"])
    return pd.DataFrame([
        {"Game": 1, "Type": label, "Team A": subdf.loc[0, "Team"], "Team B": subdf.loc[3, "Team"]},
        {"Game": 2, "Type": label, "Team A": subdf.loc[1, "Team"], "Team B": subdf.loc[2, "Team"]},
    ])

first_four = pd.concat(
    [make_matchups(lowest_4_aq, "AQ (16-seed play-in)"), make_matchups(lowest_4_al, "At-Large (play-in)")],
    ignore_index=True
)

# -----------------------------
# VISUALS
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Field Size", FIELD_SIZE)
c2.metric("AQs", AQ_COUNT)
c3.metric("At-Larges", AT_LARGE_COUNT)
c4.metric("Top Conference", top_conf)

st.markdown("## Conference Strength (avg Statistical Strength; LOWER = stronger)")
st.dataframe(
    conf_strength[["Conference", "Conf_StatStrength_Avg", "Conf_Rank", "AtLarge_WinPct_Threshold"]]
    .sort_values("Conf_Rank"),
    use_container_width=True,
    height=420
)

st.markdown("## Automatic Qualifiers (AQ)")
aq_df = pd.DataFrame({"Conference": list(aq_map.keys()), "AQ Team": list(aq_map.values())})
aq_df = aq_df.merge(conf_strength[["Conference", "Conf_Rank"]], on="Conference", how="left").sort_values("Conf_Rank")
st.dataframe(aq_df, use_container_width=True, height=450)

st.markdown("## First Four")
st.caption("4 lowest-seeded AQs play for 16-seed slots; 4 lowest-seeded at-larges play in.")
st.dataframe(first_four, use_container_width=True, height=160)

st.markdown("## Bubble Buckets")
b1, b2, b3, b4 = st.columns(4)

def bucket_table(title: str, teams: list[str]):
    sub = df[df["Team"].isin(teams)][[
        "Team", "Conference",
        "WIN_PCT", "AtLarge_Eligible", "AtLarge_WinPct_Threshold",
        "SOS_TierValue", "SOS_TierPoints",
        "Statistical_Strength", "Historical_Value", "Championship_Criteria",
        "Conf_TeamScore", "AtLarge_Score"
    ]].sort_values("AtLarge_Score", ascending=False)
    st.write(f"**{title}**")
    st.dataframe(sub, use_container_width=True, height=300)

with b1: bucket_table("Last 4 Byes", l4b)
with b2: bucket_table("Last 4 In", l4i)
with b3: bucket_table("First 4 Out", f4o)
with b4: bucket_table("Next 4 Out", n4o)

st.markdown("## Full Projected Field (Seeded)")
show_cols = [
    "Overall_Rank", "Seed_Line", "Team", "Conference", "Is_AQ",
    "FirstFour_Type",
    "WIN_PCT", "AtLarge_Eligible", "AtLarge_WinPct_Threshold",
    "SOS_TierValue", "SOS_TierPoints",
    "Statistical_Strength", "Historical_Value", "Championship_Criteria",
    "Conf_TeamScore", "AtLarge_Score", "Seed_Score", "Bubble_Bucket"
]

out = field_df.merge(
    df[["Team", "Bubble_Bucket", "AtLarge_Eligible", "AtLarge_WinPct_Threshold", "AtLarge_Score"]],
    on="Team",
    how="left"
)

st.dataframe(
    out[show_cols].sort_values(["Seed_Line", "Seed_Score"], ascending=[True, False]),
    use_container_width=True,
    height=780
)

st.markdown("## At-Large Board (Top 120)")
st.dataframe(
    atl_board.head(120)[[
        "ATL_Rank", "Team", "Conference",
        "WIN_PCT", "AtLarge_Eligible", "AtLarge_WinPct_Threshold",
        "SOS_TierValue", "SOS_TierPoints",
        "Statistical_Strength", "Historical_Value", "Championship_Criteria",
        "Conf_TeamScore", "AtLarge_Score", "Bubble_Bucket"
    ]],
    use_container_width=True,
    height=780
)

st.markdown("## Bids by Conference")
bids = (
    df[df["In_Field"]]
    .groupby("Conference", as_index=False)["Team"]
    .count()
    .rename(columns={"Team": "Bids"})
    .merge(conf_strength[["Conference", "Conf_Rank"]], on="Conference", how="left")
    .sort_values(["Bids", "Conf_Rank"], ascending=[False, True])
)
st.bar_chart(bids.set_index("Conference")["Bids"])
