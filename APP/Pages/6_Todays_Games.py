# APP/Pages/6_Todays_Games.py
"""
Today's Games

What this page does
-------------------
- Loads today's schedule rows from: Data/26_March_Madness_Databook/2026 Schedule Transfer-Table 1.csv
- Loads historical game rows from: Data/26_March_Madness_Databook/Daily_predictor_data-Table 1.csv
- Trains a **single multi-output** RandomForestRegressor to predict [Points, Opp Points] using:
    • ALL numeric columns in the daily predictor (except the target columns "Points" and "Opp Points")
    • PLUS categorical encodings for: Team, Opponent, Coach Name, Opponent Coach, Conference, Opponent Conference
    • PLUS simple binary flags: HAN(Home/Away/Neutral) → Home flag, Non Conference Game flag (if present)

- Builds prediction feature vectors for each **unique** scheduled matchup (dedup mirrored pairs)
    • Uses the SAME feature names used at training time
    • Where a feature is not present on a schedule row, fills with 0 (ranks treated as 51)
    • Categorical values are encoded with the same fitted OrdinalEncoder (unknowns encoded -1)

- Sorts the list of matchups by priority:
    1) Top 25 Opponent (True first)
    2) March Madness Opponent (True first)
    3) Opponent power-conference (SEC, Big Ten, Big 12, ACC)
    4) Date (ascending)

- Displays each matchup with an expander that shows:
    • Team / Opponent tables: Conference, Coach, Wins, Losses (when available)
    • Separate “Top 50” tables for each team:
        – Any columns in that team’s schedule-row that look like rank columns (ends with “RANK”, “_Rank”, “ Rank”, case-insensitive)
          and have value 1–50 are included
    • If the mirrored schedule row exists in the file (it usually does), opponent top-50 is pulled from that mirrored row

Key details & assumptions
-------------------------
- Dates in the schedule are **MM/DD/YYYY**. We parse with `format="%m/%d/%Y"` (no dayfirst).
- We treat missing numeric rank-like fields as **51** (so they won’t appear in the Top-50 tables).
- We treat other missing numeric features as **0** for modeling.
- NAs in categorical columns are treated as a string "NA".
- This page **re-trains** the model each run using the daily predictor file.
- No confidence intervals, just predicted points for and against.
- No external “All Stats” file is used here.

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

def force_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def interpret_han(v: Any) -> Optional[str]:
    """Map HAN-like strings to {'Home','Away','Neutral'} or None if unclear."""
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    # clean some dashes or weirdness
    s = s.replace("-", "").replace("_", "").replace(" ", "")
    if s in ("H", "HOME"):
        return "Home"
    if s in ("A", "AWAY"):
        return "Away"
    if s in ("N", "NEUTRAL"):
        return "Neutral"
    # broader tests
    raw = str(v).strip().upper()
    if "NEUTRAL" in raw:
        return "Neutral"
    if "HOME" in raw and "AWAY" not in raw:
        return "Home"
    if "AWAY" in raw and "HOME" not in raw:
        return "Away"
    return None

def is_rank_column(col_name: str) -> bool:
    """Heuristic: determine if a column name looks like a 'rank' column."""
    if col_name is None:
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
    """Interpret typical 'true' indicators in csvs."""
    if pd.isna(x):
        return False
    s = str(x).strip().upper()
    if s in ("1", "TRUE", "YES", "Y"):
        return True
    try:
        # allow numeric > 0
        return float(s) > 0
    except Exception:
        return False


# =============================================================================
# LOAD DATA — with robust cleaning and NA handling
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
# CLEAN / PREP: schedule_df
# =============================================================================

schedule_df = schedule_df_raw.copy()

# Date parsing: schedule is MM/DD/YYYY guaranteed by user
date_col = find_col(schedule_df, ["Date", "date", "Game_Date", "Game Date"])
if date_col is None:
    # if not present, assume first column might be date (edge case)
    date_col = schedule_df.columns[0]

# Use exact format to avoid 03/11 vs 11/03 confusion
schedule_df["__Date_parsed"] = pd.to_datetime(
    schedule_df[date_col].astype(str).str.strip(),
    format="%m/%d/%Y",
    errors="coerce"
)

# Drop rows without valid dates
schedule_df = schedule_df.dropna(subset=["__Date_parsed"]).reset_index(drop=True)

# Ensure we have the canonical team/opponent columns
TEAM_COL = find_col(schedule_df, ["Team", "Teams", "team", "Home", "Home Team"])
OPP_COL  = find_col(schedule_df, ["Opponent", "Opp", "opponent", "Away", "Away Team"])
if TEAM_COL is None or OPP_COL is None:
    st.error("Could not find 'Team' and 'Opponent' columns in the schedule file.")
    st.stop()

# Optional columns for HAN, NonConf, Top25, MM, conferences and coaches, wins/losses
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

# Build a fast lookup of mirrored rows for Top-50 extraction / opponent meta
# Key is (team_lower, opp_lower, date.date())
def make_key(team: str, opp: str, date_val: pd.Timestamp) -> Tuple[str, str, Optional[datetime.date]]:
    tl = (team or "").strip().lower()
    ol = (opp or "").strip().lower()
    d  = date_val.date() if isinstance(date_val, pd.Timestamp) and not pd.isna(date_val) else None
    # Use ordered triple to avoid sorting errors; still unique for pair+date
    return (tl, ol, d)

schedule_idx = {}
for i, r in schedule_df.iterrows():
    k = make_key(str(r[TEAM_COL]), str(r[OPP_COL]), r["__Date_parsed"])
    schedule_idx[k] = i  # last seen wins; OK

# Rank-like columns in the schedule file (used for Top-50)
rank_columns = [c for c in schedule_df.columns if is_rank_column(c)]

# Replace NAs:
# - For numeric rank-like columns: 51 (so they won't show in top-50)
# - For other numeric: 0
# - For everything else: keep NA for now or fill with "NA" for categoricals later
for c in schedule_df.columns:
    if pd.api.types.is_numeric_dtype(schedule_df[c]):
        if is_rank_column(c):
            schedule_df[c] = schedule_df[c].fillna(51)
        else:
            schedule_df[c] = schedule_df[c].fillna(0.0)


# =============================================================================
# CLEAN / PREP: daily_df (TRAINING)
# =============================================================================

daily_df = daily_df_raw.copy()

# Targets: "Points", "Opp Points" (names may vary slightly)
POINTS_COL     = find_col(daily_df, ["Points", "points", "PTS", "Team Points"])
OPP_POINTS_COL = find_col(daily_df, ["Opp Points", "OppPoints", "Opp_Points", "OPP PTS", "OPP_PTS"])

if POINTS_COL is None or OPP_POINTS_COL is None:
    st.error("Daily predictor file must include 'Points' and 'Opp Points' columns.")
    st.stop()

# Fill numeric ranks → 51, other numeric → 0
for c in daily_df.columns:
    if pd.api.types.is_numeric_dtype(daily_df[c]):
        if is_rank_column(c):
            daily_df[c] = daily_df[c].fillna(51)
        else:
            daily_df[c] = daily_df[c].fillna(0.0)

# We will train on:
#  - all numeric columns EXCEPT the two targets
#  - categorical encodings for: Team, Opponent, Coach Name, Opponent Coach, Conference, Opponent Conference
CAT_COLS = [
    find_col(daily_df, ["Team", "Teams", "team"]),
    find_col(daily_df, ["Opponent", "Opp", "opponent"]),
    find_col(daily_df, ["Coach Name", "Coach", "Coach_Name"]),
    find_col(daily_df, ["Opponent Coach", "Opp Coach"]),
    find_col(daily_df, ["Conference", "Conf", "conference"]),
    find_col(daily_df, ["Opponent Conference", "Opp Conference"]),
]
# Filter out None
CAT_COLS = [c for c in CAT_COLS if c is not None]

# Create flags if present
DAILY_HAN_COL     = find_col(daily_df, ["HAN", "Han", "Home/Away", "Location Type", "HomeAway"])
DAILY_NONCONF_COL = find_col(daily_df, ["Non Conference Game", "NonConference", "Non Conf", "NonConf", "Non-Conference"])

# Prepare training features (numeric base)
numeric_cols_all = daily_df.select_dtypes(include=[np.number]).columns.tolist()
# Remove the target columns from numeric features if present
numeric_feature_cols = [c for c in numeric_cols_all if c not in (POINTS_COL, OPP_POINTS_COL)]

# Prepare y (2 targets)
y_points = pd.to_numeric(daily_df[POINTS_COL], errors="coerce")
y_opp    = pd.to_numeric(daily_df[OPP_POINTS_COL], errors="coerce")
target_mask = (~y_points.isna()) & (~y_opp.isna())

if target_mask.sum() == 0:
    st.error("No historical rows in daily predictor have both 'Points' and 'Opp Points'.")
    st.stop()

# Prepare X numeric matrix
X_num = daily_df.loc[target_mask, numeric_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

# Binary extra flags (if present)
extra_flags = []
if DAILY_HAN_COL is not None:
    han_flag_series = daily_df.loc[target_mask, DAILY_HAN_COL].apply(lambda v: 1.0 if interpret_han(v) == "Home" else 0.0)
    extra_flags.append(("__flag_home", han_flag_series))
if DAILY_NONCONF_COL is not None:
    nonconf_flag_series = daily_df.loc[target_mask, DAILY_NONCONF_COL].apply(lambda v: 1.0 if boolish(v) else 0.0)
    extra_flags.append(("__flag_nonconf", nonconf_flag_series))

for name, series in extra_flags:
    X_num[name] = series.astype(float).fillna(0.0)

# Ordinal-encode categorical columns (handle_unknown set to -1)
enc = None
X_cat = None
if CAT_COLS:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_mat = daily_df.loc[target_mask, CAT_COLS].fillna("NA").astype(str).values
    try:
        X_cat = enc.fit_transform(cat_mat)
    except Exception:
        X_cat = None

# Combine numeric + categorical into final X_train
if X_cat is not None:
    X_train = np.hstack([X_num.values, X_cat])
else:
    X_train = X_num.values

# Final Y (multi-output)
Y_train = np.column_stack([
    y_points.loc[target_mask].values,
    y_opp.loc[target_mask].values
])

# Train model
st.write(f"Training multi-output RandomForestRegressor on {X_train.shape[0]} rows, {X_train.shape[1]} features…")
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, Y_train)
st.write("Model trained.")


# =============================================================================
# FEATURE PIPELINE FOR PREDICTION (SCHEDULE ROW → feature vector)
# =============================================================================

# Keep an ordered list of feature names for alignment at prediction time
feature_names_numeric = list(X_num.columns)  # numeric + extra flags from daily (names preserved)
cat_cols_for_inference = list(CAT_COLS)      # the exact cat col list used at training

def schedule_row_to_numeric_features(row: pd.Series) -> np.ndarray:
    """
    Create the numeric portion of the feature vector for a schedule row.
    We try to read **exactly** the training numeric_feature_cols (by name).
    If a numeric_feature_col is missing on the schedule row, we supply 0 (or 51 for rank-like).
    """
    vals = []
    for col in feature_names_numeric:
        if col in row.index:
            v = row.get(col)
            # If the column appears rank-like, treat NA as 51 (though we already filled schedule NAs)
            default_val = 51.0 if is_rank_column(col) else 0.0
            vals.append(to_num(v, default=default_val))
        else:
            # The schedule doesn’t have this exact numeric feature — give 0 (or 51 for ranks)
            default_val = 51.0 if is_rank_column(col) else 0.0
            vals.append(default_val)
    return np.asarray(vals, dtype=float)

def schedule_row_to_categorical_features(row: pd.Series) -> Optional[np.ndarray]:
    """
    Transform schedule row categorical columns using the fitted OrdinalEncoder.
    Uses the **same** CAT_COLS as training. Unknown values map to -1 automatically.
    """
    if enc is None or not cat_cols_for_inference:
        return None
    row_vals = []
    for col in cat_cols_for_inference:
        if col in row.index:
            row_vals.append(str(row.get(col)) if pd.notna(row.get(col)) else "NA")
        else:
            # try symmetric names for opponent fields if needed (rare)
            if "Opponent" in col and (alt := col.replace("Opponent ", "")) in row.index:
                row_vals.append(str(row.get(alt)) if pd.notna(row.get(alt)) else "NA")
            else:
                row_vals.append("NA")
    try:
        enc_vec = enc.transform([row_vals])  # shape (1, k)
        return enc_vec[0, :]
    except Exception:
        return np.array([-1] * len(cat_cols_for_inference), dtype=float)

def build_feature_vec_for_matchup(row: pd.Series) -> np.ndarray:
    """
    Construct the full feature vector for a schedule row using:
      - numeric features (aligned to training numeric feature list)
      - categorical encodings aligned to CAT_COLS
    """
    num_part = schedule_row_to_numeric_features(row)
    cat_part = schedule_row_to_categorical_features(row)
    if cat_part is None:
        return num_part
    return np.hstack([num_part, cat_part]).astype(float)


# =============================================================================
# DEDUP MATCHUPS (avoid listing mirrored rows twice)
# =============================================================================

def priority_for_row(row: pd.Series) -> Tuple[int, int, int, pd.Timestamp]:
    """
    Sorting priority: Top 25 → March Madness → Power conf → Date
    Lower is earlier in the list.
    """
    # Top-25 flag
    is_top25 = False
    if TOP25_COL and TOP25_COL in row.index:
        is_top25 = boolish(row.get(TOP25_COL))

    # MM flag
    is_mm = False
    if MM_COL and MM_COL in row.index:
        is_mm = boolish(row.get(MM_COL))

    # Opponent conference priority
    pow_conf_set = {"SEC", "BIG TEN", "B1G", "BIG 12", "BIG12", "ACC"}
    opp_conf_val = ""
    if OPP_CONF_COL and OPP_CONF_COL in row.index:
        opp_conf_val = str(row.get(OPP_CONF_COL) or "").strip().upper()
    conf_pri = 9
    if opp_conf_val in pow_conf_set:
        conf_pri = 1
    # tie-breaker by date
    return (0 if is_top25 else 1, 0 if is_mm else 1, conf_pri, row["__Date_parsed"])

# Build unique list of matchups
seen = set()
unique_rows = []
for i, r in schedule_df.iterrows():
    team = str(r[TEAM_COL]) if pd.notna(r[TEAM_COL]) else ""
    opp  = str(r[OPP_COL]) if pd.notna(r[OPP_COL]) else ""
    d    = r["__Date_parsed"]
    # Dedup mirrored pairs (unordered)
    unordered_pair = (min(team.lower(), opp.lower()), max(team.lower(), opp.lower()), d.date() if not pd.isna(d) else None)
    if unordered_pair in seen:
        continue
    seen.add(unordered_pair)
    unique_rows.append((priority_for_row(r), i))

# Sort by priority
unique_rows.sort(key=lambda x: x[0])
sorted_indices = [idx for _, idx in unique_rows]

if len(sorted_indices) == 0:
    st.info("No scheduled games found.")
    st.stop()


# =============================================================================
# PREDICTION LOOP
# =============================================================================

pred_records = []
for idx in sorted_indices:
    row = schedule_df.iloc[idx]

    # Build feature vector aligned to the training features
    X_vec = build_feature_vec_for_matchup(row).reshape(1, -1)

    # Ensure input width matches training width
    if X_vec.shape[1] != X_train.shape[1]:
        # Align by truncating or zero-padding on the right
        aligned = np.zeros((1, X_train.shape[1]), dtype=float)
        min_cols = min(X_vec.shape[1], X_train.shape[1])
        aligned[0, :min_cols] = X_vec[0, :min_cols]
        X_vec = aligned

    # Predict [Points, Opp Points]
    try:
        y_pred = rf.predict(X_vec)[0]
    except Exception:
        y_pred = np.array([0.0, 0.0])

    team_name = str(row[TEAM_COL]).strip()
    opp_name  = str(row[OPP_COL]).strip()

    # Metadata for left/right team tables
    def safe_get(row_: pd.Series, colname: Optional[str]) -> Any:
        if colname and colname in row_.index:
            return row_.get(colname)
        return ""

    # Attempt to fetch the mirrored row (opponent as "home" team in file) for opponent meta and top-50
    mirror_key = make_key(opp_name, team_name, row["__Date_parsed"])
    mirror_row = schedule_df.iloc[schedule_idx[mirror_key]] if mirror_key in schedule_idx else None

    # Team meta
    team_meta = {
        "Coach": safe_get(row, COACH_COL),
        "Conference": safe_get(row, CONF_COL),
        "Wins": safe_get(row, WINS_COL),
        "Losses": safe_get(row, LOSSES_COL),
    }
    # Opponent meta (prefer mirror row if available)
    if mirror_row is not None:
        opp_meta = {
            "Coach": safe_get(mirror_row, COACH_COL) if COACH_COL else safe_get(row, OPP_COACH_COL),
            "Conference": safe_get(mirror_row, CONF_COL) if CONF_COL else safe_get(row, OPP_CONF_COL),
            "Wins": safe_get(mirror_row, WINS_COL),
            "Losses": safe_get(mirror_row, LOSSES_COL),
        }
        row_for_opp_ranks = mirror_row
    else:
        opp_meta = {
            "Coach": safe_get(row, OPP_COACH_COL),
            "Conference": safe_get(row, OPP_CONF_COL),
            "Wins": "",
            "Losses": "",
        }
        row_for_opp_ranks = row  # fallback

    # Top-50 list extraction
    def extract_top50_from_row(r_: pd.Series) -> List[Tuple[str, int]]:
        out = []
        for c in rank_columns:
            try:
                val = to_int(r_.get(c), default=9999)
                if 1 <= val <= 50:
                    out.append((c, val))
            except Exception:
                continue
        # Sort by rank ascending
        out.sort(key=lambda z: z[1])
        return out

    team_top50 = extract_top50_from_row(row)
    opp_top50  = extract_top50_from_row(row_for_opp_ranks)

    pred_records.append({
        "Date": row["__Date_parsed"],
        "Team": team_name,
        "Opponent": opp_name,
        "Pred_Team_Points": int(round(max(0.0, y_pred[0]))),
        "Pred_Opp_Points": int(round(max(0.0, y_pred[1]))),
        "Priority": priority_for_row(row),
        "Team_Meta": team_meta,
        "Opp_Meta": opp_meta,
        "Team_Top50": team_top50,
        "Opp_Top50": opp_top50
    })


# =============================================================================
# FILTERS / CONTROLS
# =============================================================================

st.markdown("---")
st.subheader("Filters")

# Show only Top-25 games
show_top25_only = st.checkbox("Show only Top-25 opponent games", value=False)

# Optional opponent-conference filter list
opponent_conf_series = schedule_df[OPP_CONF_COL] if OPP_CONF_COL else pd.Series([], dtype=object)
opponent_conf_opts = sorted(list({str(x) for x in opponent_conf_series.dropna().unique().tolist()}))
conf_multiselect = st.multiselect("Filter by opponent conference", opponent_conf_opts, default=[])

# Apply filters
def pass_filters(rec: Dict[str, Any]) -> bool:
    # Find a schedule row to read Top25/Conference flags for filter evaluation
    k = make_key(rec["Team"], rec["Opponent"], rec["Date"])
    base_row = schedule_df.iloc[schedule_idx[k]] if k in schedule_idx else None
    if base_row is None:
        return True  # if we can't locate row, don't filter it out

    # Top-25 filter
    if show_top25_only and TOP25_COL and TOP25_COL in base_row.index:
        if not boolish(base_row.get(TOP25_COL)):
            return False

    # Conference multiselect
    if conf_multiselect and OPP_CONF_COL and OPP_CONF_COL in base_row.index:
        oc = str(base_row.get(OPP_CONF_COL) or "").strip()
        if oc not in conf_multiselect:
            return False

    return True

pred_records_filtered = [r for r in pred_records if pass_filters(r)]

if not pred_records_filtered:
    st.info("No games match the selected filters.")
    st.stop()


# =============================================================================
# DISPLAY
# =============================================================================

st.markdown("---")
st.subheader("Predicted Games (sorted by priority)")

for rec in pred_records_filtered:
    date_str = rec["Date"].strftime("%m/%d/%Y") if not pd.isna(rec["Date"]) else "TBD"
    game_header = f"{rec['Team']} vs {rec['Opponent']} — {date_str} — Pred: {rec['Pred_Team_Points']} - {rec['Pred_Opp_Points']}"

    with st.expander(game_header, expanded=False):

        # -------------------------
        # Meta tables (Team / Opponent)
        # -------------------------
        st.markdown("#### Teams")

        left, right = st.columns(2)

        with left:
            st.write(f"**{rec['Team']}**")
            st.write(f"Conference: {rec['Team_Meta'].get('Conference', '')}")
            st.write(f"Coach: {rec['Team_Meta'].get('Coach', '')}")
            st.write(f"Wins: {rec['Team_Meta'].get('Wins', '')}")
            st.write(f"Losses: {rec['Team_Meta'].get('Losses', '')}")

        with right:
            st.write(f"**{rec['Opponent']}**")
            st.write(f"Conference: {rec['Opp_Meta'].get('Conference', '')}")
            st.write(f"Coach: {rec['Opp_Meta'].get('Coach', '')}")
            st.write(f"Wins: {rec['Opp_Meta'].get('Wins', '')}")
            st.write(f"Losses: {rec['Opp_Meta'].get('Losses', '')}")

        st.markdown("---")

        # -------------------------
        # Top 50 lists (split per team)
        # -------------------------
        st.markdown("#### Top 50 in the Nation — Category and Rank")

        c1, c2 = st.columns(2)

        def _df_top50(lst: List[Tuple[str, int]]) -> pd.DataFrame:
            if not lst:
                return pd.DataFrame([{"Category": "None in Top 50", "Rank": ""}])
            return pd.DataFrame([{"Category": k, "Rank": v} for k, v in lst])

        with c1:
            st.write(f"**{rec['Team']}**")
            st.dataframe(_df_top50(rec["Team_Top50"]), use_container_width=True)

        with c2:
            st.write(f"**{rec['Opponent']}**")
            st.dataframe(_df_top50(rec["Opp_Top50"]), use_container_width=True)

        st.markdown("---")
        st.write("Notes:")
        st.write("- Predictions are produced by a single multi-output RandomForestRegressor trained on your Daily predictor file.")
        st.write("- Date parsing is MM/DD/YYYY.")
        st.write("- Rank-based columns are treated as 51 when missing so they don’t appear in the Top-50 lists.")
        st.write("- Categorical features (team, opponent, coach names, conferences) are encoded to ensure matchup-specific variation.")

# -----------------------------------------------------------------------------
# CSV Download
# -----------------------------------------------------------------------------
st.markdown("---")
out_df = pd.DataFrame([{
    "Date": r["Date"].strftime("%Y-%m-%d") if not pd.isna(r["Date"]) else "",
    "Team": r["Team"],
    "Opponent": r["Opponent"],
    "Pred_Team_Points": r["Pred_Team_Points"],
    "Pred_Opp_Points": r["Pred_Opp_Points"]
} for r in pred_records_filtered])

csv_bytes = out_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download predictions CSV",
    data=csv_bytes,
    file_name="todays_games_predictions.csv",
    mime="text/csv"
)

# =============================================================================
# END OF FILE
# =============================================================================
