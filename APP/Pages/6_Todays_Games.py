# APP/Pages/6_Todays_Games.py
"""
Today's Games — Single Multi-Output Model (improved featurization)

What this page does
-------------------
- Loads the schedule from:
    Data/26_March_Madness_Databook/2026 Schedule Transfer-Table 1.csv
- Loads historical games from:
    Data/26_March_Madness_Databook/Daily_predictor_data-Table 1.csv
- Trains ONE multi-output RandomForestRegressor to predict [Points, Opp Points].

Model features
--------------
- ALL numeric columns from the daily file EXCEPT the two targets.
  • Numeric rank-like columns (name looks like "*rank*") → NA filled with **51**.
  • Other numerics → NA filled with **0**.
- Binary flags engineered and appended to numeric features (and therefore part of the same feature space):
  • __flag_home              — 1 if HAN indicates "Home", else 0
  • __flag_nonconf           — 1 if non-conference game, else 0
  • __flag_team_power_conf   — 1 if team conf in {SEC, BIG TEN, BIG 12, ACC}
  • __flag_opp_power_conf    — 1 if opp conf in {SEC, BIG TEN, BIG 12, ACC}
- Categorical encodings (OrdinalEncoder; unknowns → -1):
  • Team, Opponent, Coach Name, Opponent Coach, Conference, Opponent Conference

Prediction pipeline
-------------------
- Builds schedule feature vectors that EXACTLY align to the training feature list.
- For columns missing on the schedule:
  • If the feature name looks like a rank → uses 51
  • Else uses 0
- Special-cases the engineered flags by computing them from the schedule row if those
  names (e.g. "__flag_home") appear in the training feature list.
- Deduplicates mirrored rows (Albany vs Marquette vs Marquette vs Albany on same date).
- Sort priority: Top-25 first → March Madness opponent → Power-conf opponent → Date.
- Each matchup renders with an expander containing metadata and SEPARATE "Top 50" lists
  for each team (categories with rank ≤ 50 from that row / mirrored row).

Notes
-----
- Date parsing uses fixed US format "%m/%d/%Y".
- This page re-trains the model on every run so it adapts as Daily file grows.
- No confidence intervals are shown; just predicted scores.
"""

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = "Data/26_March_Madness_Databook"
SCHEDULE_PATH = os.path.join(BASE_DIR, "2026 Schedule Transfer-Table 1.csv")
DAILY_PATH    = os.path.join(BASE_DIR, "Daily_predictor_data-Table 1.csv")

PAGE_TITLE = "Today's Games — Predicted Scores"
st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title(PAGE_TITLE)

POWER_CONFS = {"SEC", "BIG TEN", "B1G", "BIG 12", "BIG12", "ACC"}


# =============================================================================
# UTILS
# =============================================================================

def col_exists(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def interpret_han(v: Any) -> Optional[str]:
    """Map HAN-like strings to {'Home','Away','Neutral'} or None if unclear."""
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    s_compact = s.replace("-", "").replace("_", "").replace(" ", "")
    if s_compact in ("H", "HOME"):
        return "Home"
    if s_compact in ("A", "AWAY"):
        return "Away"
    if s_compact in ("N", "NEUTRAL"):
        return "Neutral"
    if "NEUTRAL" in s:
        return "Neutral"
    if "HOME" in s and "AWAY" not in s:
        return "Home"
    if "AWAY" in s and "HOME" not in s:
        return "Away"
    return None

def is_rank_column(col_name: str) -> bool:
    """Heuristic: determine if a column name looks like a 'rank' column."""
    if not col_name:
        return False
    u = col_name.upper().strip()
    return (
        "RANK" in u
        or u.endswith("_RANK")
        or u.endswith(" RANK")
        or u.endswith("_RANKING")
        or u.endswith(" RANKING")
    )

def to_num(x: Any, default: float = 0.0) -> float:
    try:
        v = float(pd.to_numeric(x, errors="coerce"))
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default

def to_int(x: Any, default: int = 0) -> int:
    v = to_num(x, default=np.nan)
    if pd.isna(v):
        return default
    try:
        return int(round(v))
    except Exception:
        return default

def boolish(x: Any) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip().upper()
    if s in ("1", "TRUE", "YES", "Y"):
        return True
    try:
        return float(s) > 0
    except Exception:
        return False

def power_conf_flag(conf_val: Any) -> float:
    if pd.isna(conf_val):
        return 0.0
    c = str(conf_val).strip().upper()
    return 1.0 if c in POWER_CONFS else 0.0


# =============================================================================
# LOAD DATA (safe)
# =============================================================================

@st.cache_data
def load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="latin1")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Failed to read {path}: {e}")
        return None

schedule_df_raw = load_csv(SCHEDULE_PATH)
daily_df_raw    = load_csv(DAILY_PATH)

if schedule_df_raw is None:
    st.error(f"Schedule file not found at: {SCHEDULE_PATH}")
    st.stop()
if daily_df_raw is None:
    st.error(f"Daily predictor file not found at: {DAILY_PATH}")
    st.stop()

st.info("CSV files loaded.")


# =============================================================================
# PREP SCHEDULE
# =============================================================================

schedule_df = schedule_df_raw.copy()

# Date parsing: fixed US format MM/DD/YYYY
DATE_COL = find_col(schedule_df, ["Date", "date", "Game_Date", "Game Date"]) or schedule_df.columns[0]
schedule_df["__Date_parsed"] = pd.to_datetime(
    schedule_df[DATE_COL].astype(str).str.strip(),
    format="%m/%d/%Y",
    errors="coerce"
)
schedule_df = schedule_df.dropna(subset=["__Date_parsed"]).reset_index(drop=True)

TEAM_COL      = find_col(schedule_df, ["Team", "Teams", "team", "Home", "Home Team"])
OPP_COL       = find_col(schedule_df, ["Opponent", "Opp", "opponent", "Away", "Away Team"])
HAN_COL       = find_col(schedule_df, ["HAN", "Han", "Home/Away", "Location Type", "HomeAway"])
NONCONF_COL   = find_col(schedule_df, ["Non Conference Game", "NonConference", "Non Conf", "NonConf", "Non-Conference"])
TOP25_COL     = find_col(schedule_df, ["Top 25 Opponent", "Top25", "Top 25", "TOP25"])
MM_COL        = find_col(schedule_df, ["March Madness Opponent", "March Madness", "March_Madness"])
CONF_COL      = find_col(schedule_df, ["Conference", "Conf", "conference"])
OPP_CONF_COL  = find_col(schedule_df, ["Opponent Conference", "Opp Conference"])
COACH_COL     = find_col(schedule_df, ["Coach Name", "Coach", "Coach_Name"])
OPP_COACH_COL = find_col(schedule_df, ["Opponent Coach", "Opp Coach"])
WINS_COL      = find_col(schedule_df, ["Wins", "wins"])
LOSSES_COL    = find_col(schedule_df, ["Losses", "losses"])

if TEAM_COL is None or OPP_COL is None:
    st.error("Could not find Team and Opponent columns in the schedule file.")
    st.stop()

# Fill numeric columns: ranks → 51, others → 0
for c in schedule_df.columns:
    if pd.api.types.is_numeric_dtype(schedule_df[c]):
        schedule_df[c] = schedule_df[c].fillna(51 if is_rank_column(c) else 0.0)

# Map rows for mirrored lookup (opp as team on same date)
def sched_key(team: str, opp: str, d: pd.Timestamp) -> Tuple[str, str, Optional[datetime.date]]:
    tl = (team or "").strip().lower()
    ol = (opp or "").strip().lower()
    dt = d.date() if isinstance(d, pd.Timestamp) and pd.notna(d) else None
    return (tl, ol, dt)

schedule_row_index = {}
for i, r in schedule_df.iterrows():
    schedule_row_index[sched_key(str(r[TEAM_COL]), str(r[OPP_COL]), r["__Date_parsed"])] = i

# Gather all schedule rank columns (used later for top-50 lists)
sched_rank_cols = [c for c in schedule_df.columns if is_rank_column(c)]


# =============================================================================
# PREP DAILY (TRAINING)
# =============================================================================

daily_df = daily_df_raw.copy()

POINTS_COL     = find_col(daily_df, ["Points", "points", "PTS", "Team Points"])
OPP_POINTS_COL = find_col(daily_df, ["Opp Points", "OppPoints", "Opp_Points", "OPP PTS", "OPP_PTS"])
if POINTS_COL is None or OPP_POINTS_COL is None:
    st.error("Daily predictor must include 'Points' and 'Opp Points' columns.")
    st.stop()

# Fill numerics similarly to schedule
for c in daily_df.columns:
    if pd.api.types.is_numeric_dtype(daily_df[c]):
        daily_df[c] = daily_df[c].fillna(51 if is_rank_column(c) else 0.0)

# Categorical columns to encode
CAT_COLS = [
    find_col(daily_df, ["Team", "Teams", "team"]),
    find_col(daily_df, ["Opponent", "Opp", "opponent"]),
    find_col(daily_df, ["Coach Name", "Coach", "Coach_Name"]),
    find_col(daily_df, ["Opponent Coach", "Opp Coach"]),
    find_col(daily_df, ["Conference", "Conf", "conference"]),
    find_col(daily_df, ["Opponent Conference", "Opp Conference"]),
]
CAT_COLS = [c for c in CAT_COLS if c is not None]

# Targets and mask
y_points = pd.to_numeric(daily_df[POINTS_COL], errors="coerce")
y_opp    = pd.to_numeric(daily_df[OPP_POINTS_COL], errors="coerce")
target_mask = (~y_points.isna()) & (~y_opp.isna())

if target_mask.sum() == 0:
    st.error("No rows in the daily file have both Points and Opp Points.")
    st.stop()

# Base numeric features (all daily numerics except the two targets)
all_daily_num_cols = daily_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_feature_cols = [c for c in all_daily_num_cols if c not in (POINTS_COL, OPP_POINTS_COL)]

# Build X_num matrix
X_num = daily_df.loc[target_mask, numeric_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# Engineered binary flags (these will become columns in X_num with fixed names)
DAILY_HAN_COL     = find_col(daily_df, ["HAN", "Han", "Home/Away", "Location Type", "HomeAway"])
DAILY_NONCONF_COL = find_col(daily_df, ["Non Conference Game", "NonConference", "Non Conf", "NonConf", "Non-Conference"])
DAILY_CONF_COL    = find_col(daily_df, ["Conference", "Conf", "conference"])
DAILY_OPP_CONF_COL= find_col(daily_df, ["Opponent Conference", "Opp Conference"])

if "__flag_home" not in X_num.columns:
    if DAILY_HAN_COL:
        X_num["__flag_home"] = daily_df.loc[target_mask, DAILY_HAN_COL].apply(lambda v: 1.0 if interpret_han(v) == "Home" else 0.0).astype(float)
    else:
        X_num["__flag_home"] = 0.0

if "__flag_nonconf" not in X_num.columns:
    if DAILY_NONCONF_COL:
        X_num["__flag_nonconf"] = daily_df.loc[target_mask, DAILY_NONCONF_COL].apply(lambda v: 1.0 if boolish(v) else 0.0).astype(float)
    else:
        X_num["__flag_nonconf"] = 0.0

if "__flag_team_power_conf" not in X_num.columns:
    if DAILY_CONF_COL:
        X_num["__flag_team_power_conf"] = daily_df.loc[target_mask, DAILY_CONF_COL].apply(power_conf_flag).astype(float)
    else:
        X_num["__flag_team_power_conf"] = 0.0

if "__flag_opp_power_conf" not in X_num.columns:
    if DAILY_OPP_CONF_COL:
        X_num["__flag_opp_power_conf"] = daily_df.loc[target_mask, DAILY_OPP_CONF_COL].apply(power_conf_flag).astype(float)
    else:
        X_num["__flag_opp_power_conf"] = 0.0

# Ordinal-encode categorized columns (stable across train/predict)
enc = None
X_cat = None
if CAT_COLS:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_mat = daily_df.loc[target_mask, CAT_COLS].fillna("NA").astype(str).values
    try:
        X_cat = enc.fit_transform(cat_mat)
    except Exception:
        X_cat = None

# Combine numeric + categorical
if X_cat is not None:
    X_train = np.hstack([X_num.values, X_cat])
else:
    X_train = X_num.values

Y_train = np.column_stack([
    y_points.loc[target_mask].values,
    y_opp.loc[target_mask].values
])

# Train model
st.write(f"Training on {X_train.shape[0]} rows and {X_train.shape[1]} features…")

rf = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=1,
    max_features="sqrt"  # fixed: "auto" deprecated → use "sqrt" or None
)
rf.fit(X_train, Y_train)
st.write("Model trained.")


# Keep an ordered list of the final training numeric feature names (including engineered flags)
feature_names_numeric = list(X_num.columns)
cat_cols_for_inference = list(CAT_COLS)  # in order


# =============================================================================
# BUILD PREDICTION VECTORS (SCHEDULE ROW → X_vec aligned to training)
# =============================================================================

def schedule_numeric_vector(row: pd.Series) -> np.ndarray:
    """
    Build numeric features for a schedule row that align EXACTLY to the training
    numeric features (feature_names_numeric). If a feature is missing on the
    schedule row:
      - If name looks like a rank → 51
      - Else → 0
    Special-case engineered flags: __flag_home / __flag_nonconf / team/opp power flags.
    """
    vals: List[float] = []

    # Precompute schedule HAN/nonconf and conf flags
    row_han = interpret_han(row.get(HAN_COL)) if HAN_COL and HAN_COL in row.index else None
    row_nonconf = boolish(row.get(NONCONF_COL)) if NONCONF_COL and NONCONF_COL in row.index else False
    row_team_conf = row.get(CONF_COL) if CONF_COL and CONF_COL in row.index else None
    row_opp_conf  = row.get(OPP_CONF_COL) if OPP_CONF_COL and OPP_CONF_COL in row.index else None

    for fname in feature_names_numeric:
        if fname == "__flag_home":
            vals.append(1.0 if row_han == "Home" else 0.0)
            continue
        if fname == "__flag_nonconf":
            vals.append(1.0 if row_nonconf else 0.0)
            continue
        if fname == "__flag_team_power_conf":
            vals.append(power_conf_flag(row_team_conf))
            continue
        if fname == "__flag_opp_power_conf":
            vals.append(power_conf_flag(row_opp_conf))
            continue

        if fname in row.index:
            raw = row.get(fname)
            default_val = 51.0 if is_rank_column(fname) else 0.0
            vals.append(to_num(raw, default=default_val))
        else:
            default_val = 51.0 if is_rank_column(fname) else 0.0
            vals.append(default_val)

    return np.asarray(vals, dtype=float)

def schedule_categorical_vector(row: pd.Series) -> Optional[np.ndarray]:
    if enc is None or not cat_cols_for_inference:
        return None
    vals = []
    for c in cat_cols_for_inference:
        if c in row.index:
            vals.append(str(row.get(c)) if pd.notna(row.get(c)) else "NA")
        else:
            # try a few symmetric alternates (rare)
            if "Opponent" in c:
                alt = c.replace("Opponent ", "")
                if alt in row.index:
                    vals.append(str(row.get(alt)) if pd.notna(row.get(alt)) else "NA")
                else:
                    vals.append("NA")
            else:
                vals.append("NA")
    try:
        return enc.transform([vals])[0, :]
    except Exception:
        return np.array([-1] * len(vals), dtype=float)

def build_schedule_X(row: pd.Series) -> np.ndarray:
    num_part = schedule_numeric_vector(row)
    cat_part = schedule_categorical_vector(row)
    if cat_part is None:
        return num_part
    return np.hstack([num_part, cat_part]).astype(float)


# =============================================================================
# DEDUP & PRIORITY
# =============================================================================

def row_priority(row: pd.Series) -> Tuple[int, int, int, pd.Timestamp]:
    # Top-25 first
    is_top25 = boolish(row.get(TOP25_COL)) if TOP25_COL and TOP25_COL in row.index else False
    # March Madness opponent second
    is_mm    = boolish(row.get(MM_COL)) if MM_COL and MM_COL in row.index else False
    # Power-conf opponent third
    opp_conf_flag = power_conf_flag(row.get(OPP_CONF_COL)) if OPP_CONF_COL and OPP_CONF_COL in row.index else 0.0
    conf_pri = 1 if opp_conf_flag == 1.0 else 9
    return (0 if is_top25 else 1, 0 if is_mm else 1, conf_pri, row["__Date_parsed"])

seen_pairs = set()
unique_rows = []
for i, r in schedule_df.iterrows():
    t = str(r[TEAM_COL]).strip()
    o = str(r[OPP_COL]).strip()
    d = r["__Date_parsed"]
    # unordered pair for dedup
    key = (min(t.lower(), o.lower()), max(t.lower(), o.lower()), d.date() if not pd.isna(d) else None)
    if key in seen_pairs:
        continue
    seen_pairs.add(key)
    unique_rows.append((row_priority(r), i))

unique_rows.sort(key=lambda x: x[0])
sorted_indices = [idx for _, idx in unique_rows]

if not sorted_indices:
    st.info("No scheduled games found.")
    st.stop()


# =============================================================================
# PREDICT
# =============================================================================

pred_rows = []
for idx in sorted_indices:
    r = schedule_df.iloc[idx]

    X_vec = build_schedule_X(r).reshape(1, -1)

    # Align width if needed (pad/truncate)
    if X_vec.shape[1] != X_train.shape[1]:
        aligned = np.zeros((1, X_train.shape[1]), dtype=float)
        use = min(X_vec.shape[1], X_train.shape[1])
        aligned[0, :use] = X_vec[0, :use]
        X_vec = aligned

    # Predict [Points, Opp Points]
    try:
        y_pred = rf.predict(X_vec)[0]
    except Exception:
        y_pred = np.array([0.0, 0.0])

    team_name = str(r[TEAM_COL]).strip()
    opp_name  = str(r[OPP_COL]).strip()

    # Find mirror for proper opponent meta and top-50 extraction
    mirror_key = (opp_name.lower(), team_name.lower(), r["__Date_parsed"].date() if not pd.isna(r["__Date_parsed"]) else None)
    mirror_row = schedule_df.iloc[schedule_row_index[mirror_key]] if mirror_key in schedule_row_index else None

    # Meta dicts
    def g(row_: pd.Series, c: Optional[str]) -> Any:
        return row_.get(c) if (row_ is not None and c and c in row_.index) else ""

    team_meta = {
        "Conference": g(r, CONF_COL),
        "Coach": g(r, COACH_COL),
        "Wins": g(r, WINS_COL),
        "Losses": g(r, LOSSES_COL)
    }
    if mirror_row is not None:
        opp_meta = {
            "Conference": g(mirror_row, CONF_COL) if CONF_COL else g(r, OPP_CONF_COL),
            "Coach": g(mirror_row, COACH_COL) if COACH_COL else g(r, OPP_COACH_COL),
            "Wins": g(mirror_row, WINS_COL),
            "Losses": g(mirror_row, LOSSES_COL)
        }
        row_for_opp_ranks = mirror_row
    else:
        opp_meta = {
            "Conference": g(r, OPP_CONF_COL),
            "Coach": g(r, OPP_COACH_COL),
            "Wins": "",
            "Losses": ""
        }
        row_for_opp_ranks = r

    # Top-50 extractors — SEPARATE for each team
    def extract_top50(r_: pd.Series) -> List[Tuple[str, int]]:
        out = []
        for c in sched_rank_cols:
            try:
                v = to_int(r_.get(c), default=9999)
                if 1 <= v <= 50:
                    out.append((c, v))
            except Exception:
                continue
        out.sort(key=lambda z: z[1])
        return out

    team_top50 = extract_top50(r)
    opp_top50  = extract_top50(row_for_opp_ranks)

    pred_rows.append({
        "Date": r["__Date_parsed"],
        "Team": team_name,
        "Opponent": opp_name,
        "Pred_Team_Points": int(round(max(0.0, y_pred[0]))),
        "Pred_Opp_Points": int(round(max(0.0, y_pred[1]))),
        "Team_Meta": team_meta,
        "Opp_Meta": opp_meta,
        "Team_Top50": team_top50,
        "Opp_Top50": opp_top50,
        "Priority": row_priority(r)
    })


# =============================================================================
# FILTERS
# =============================================================================

st.markdown("---")
st.subheader("Filters")

show_top25_only = st.checkbox("Show only Top-25 opponent games", value=False)

opp_conf_opts = []
if OPP_CONF_COL:
    opp_conf_opts = sorted(list({str(x) for x in schedule_df[OPP_CONF_COL].dropna().unique().tolist()}))
conf_selected = st.multiselect("Filter by opponent conference", opp_conf_opts, default=[])

def passes_filters(rec: Dict[str, Any]) -> bool:
    # Locate base row to inspect filter fields
    key = (rec["Team"].lower(), rec["Opponent"].lower(), rec["Date"].date() if not pd.isna(rec["Date"]) else None)
    base_row = schedule_df.iloc[schedule_row_index[key]] if key in schedule_row_index else None
    if base_row is None:
        return True
    if show_top25_only and TOP25_COL and TOP25_COL in base_row.index:
        if not boolish(base_row.get(TOP25_COL)):
            return False
    if conf_selected and OPP_CONF_COL and OPP_CONF_COL in base_row.index:
        oc = str(base_row.get(OPP_CONF_COL) or "")
        if oc not in conf_selected:
            return False
    return True

pred_rows = [r for r in pred_rows if passes_filters(r)]
if not pred_rows:
    st.info("No games match the selected filters.")
    st.stop()

# Sort by priority key already computed
pred_rows.sort(key=lambda z: z["Priority"])


# =============================================================================
# DISPLAY
# =============================================================================

st.markdown("---")
st.subheader("Predicted Games (sorted by priority)")

for rec in pred_rows:
    date_str = rec["Date"].strftime("%m/%d/%Y") if not pd.isna(rec["Date"]) else "TBD"
    header = f"{rec['Team']} vs {rec['Opponent']} — {date_str} — Pred: {rec['Pred_Team_Points']} - {rec['Pred_Opp_Points']}"
    with st.expander(header, expanded=False):

        # Team / Opponent meta
        st.markdown("#### Teams")
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**{rec['Team']}**")
            st.write(f"Conference: {rec['Team_Meta'].get('Conference', '')}")
            st.write(f"Coach: {rec['Team_Meta'].get('Coach', '')}")
            st.write(f"Wins: {rec['Team_Meta'].get('Wins', '')}")
            st.write(f"Losses: {rec['Team_Meta'].get('Losses', '')}")
        with c2:
            st.write(f"**{rec['Opponent']}**")
            st.write(f"Conference: {rec['Opp_Meta'].get('Conference', '')}")
            st.write(f"Coach: {rec['Opp_Meta'].get('Coach', '')}")
            st.write(f"Wins: {rec['Opp_Meta'].get('Wins', '')}")
            st.write(f"Losses: {rec['Opp_Meta'].get('Losses', '')}")

        st.markdown("---")
        st.markdown("#### Top 50 in the Nation — Category and Rank")

        def df_top50(lst: List[Tuple[str, int]]) -> pd.DataFrame:
            if not lst:
                return pd.DataFrame([{"Category": "None in Top 50", "Rank": ""}])
            return pd.DataFrame([{"Category": k, "Rank": v} for k, v in lst])

        tcol, ocol = st.columns(2)
        with tcol:
            st.write(f"**{rec['Team']}**")
            st.dataframe(df_top50(rec["Team_Top50"]), use_container_width=True)
        with ocol:
            st.write(f"**{rec['Opponent']}**")
            st.dataframe(df_top50(rec["Opp_Top50"]), use_container_width=True)

        st.markdown("---")
        st.write("Notes:")
        st.write("- Predictions use a single multi-output RandomForest with numeric stats, rank fields, categorical encodings (team/opponent/coach/conferences), and engineered flags.")
        st.write("- Missing rank values are treated as 51; other missing numerics are 0.")
        st.write("- HAN is used to set a home flag. Non-conference and power-conference flags are included for both sides.")
        st.write("- Results should vary per matchup even with limited early-season data thanks to categorical encodings and rank fields.")

# Download CSV
st.markdown("---")
out = pd.DataFrame([{
    "Date": r["Date"].strftime("%Y-%m-%d") if not pd.isna(r["Date"]) else "",
    "Team": r["Team"],
    "Opponent": r["Opponent"],
    "Pred_Team_Points": r["Pred_Team_Points"],
    "Pred_Opp_Points": r["Pred_Opp_Points"]
} for r in pred_rows])
st.download_button(
    "Download predictions CSV",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="todays_games_predictions.csv",
    mime="text/csv"
)
