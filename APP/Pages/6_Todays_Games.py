# APP/Pages/6_Todays_Games.py
"""
Today's Games page.

- Retrains models each run from Daily_predictor_data-Table 1.csv.
- Uses numeric columns from daily predictor as features (all numeric cols except Points/Opp Points).
- Builds prediction features for scheduled matchups from the schedule transfer CSV.
- Predicts Points and Opp Points for each matchup (no confidence intervals).
- Sorts and displays upcoming games with a per-game expander showing team details and top-50 ranks.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime

# -------------------------
# CONFIG
# -------------------------
BASE = "Data/26_March_Madness_Databook"
SCHEDULE_FILE = os.path.join(BASE, "2026 Schedule Transfer-Table 1.csv")
DAILY_FILE = os.path.join(BASE, "Daily_predictor_data-Table 1.csv")
ALL_STATS_FILE = os.path.join(BASE, "All_Stats-THE_TABLE.csv")  # only used for optional lookups if needed

st.set_page_config(page_title="Today's Games", layout="wide")
st.title("Today's Games — Predictions")

# -------------------------
# LOAD CSVS (safe)
# -------------------------
@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="latin1")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error reading {path}: {e}")
        return None

schedule_df = load_csv(SCHEDULE_FILE)
daily_df = load_csv(DAILY_FILE)

if schedule_df is None:
    st.error(f"Schedule transfer file not found at: {SCHEDULE_FILE}")
    st.stop()
if daily_df is None:
    st.error(f"Daily predictor (training) file not found at: {DAILY_FILE}")
    st.stop()

st.info("Files loaded.")

# -------------------------
# Utility helpers
# -------------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def parse_date_series(s):
    """Parse dates robustly and return datetime series."""
    parsed = pd.to_datetime(s.astype(str).str.strip(), errors="coerce", infer_datetime_format=True)
    # if a lot of NaT, try dayfirst fallback
    if parsed.isna().mean() > 0.25:
        parsed = pd.to_datetime(s.astype(str).str.strip(), errors="coerce", dayfirst=True, infer_datetime_format=True)
    return parsed

def interpret_han(v):
    if pd.isna(v):
        return None
    s = str(v).strip().upper()
    if s in ("H", "HOME"):
        return "Home"
    if s in ("A", "AWAY"):
        return "Away"
    if "NEUTRAL" in s or s == "N":
        return "Neutral"
    if "HOME" in s and "AWAY" not in s:
        return "Home"
    if "AWAY" in s and "HOME" not in s:
        return "Away"
    return None

# -------------------------
# DATE parsing & schedule cleanup
# -------------------------
# Try common date columns
date_candidates = ["Date", "date", "Game_Date", "Game Date"]
date_col = find_col(schedule_df, date_candidates)
if date_col is None:
    # fallback to first column
    date_col = schedule_df.columns[0]

schedule_df["__Date_parsed"] = parse_date_series(schedule_df[date_col])

# drop rows without date (we only want scheduled games)
schedule_df = schedule_df.dropna(subset=["__Date_parsed"]).reset_index(drop=True)
schedule_df = schedule_df.sort_values("__Date_parsed").reset_index(drop=True)

# -------------------------
# Basic schedule column detection
# -------------------------
team_col = find_col(schedule_df, ["Team", "Teams", "team", "Home", "Home Team"])
opp_col = find_col(schedule_df, ["Opponent", "Opp", "opponent", "Away", "Away Team"])
han_col = find_col(schedule_df, ["HAN", "Han", "Home/Away", "Location", "Loc", "HomeAway", "HOME/ AWAY", "HOME/AWAY"])
top25_col = find_col(schedule_df, ["Top 25 Opponent", "Top25", "Top 25", "TOP25"])
mm_col = find_col(schedule_df, ["March Madness Opponent", "March Madness", "March_Madness"])
nonconf_col = find_col(schedule_df, ["Non Conference Game", "NonConference", "Non Conf", "NonConf", "Non-Conference"])

# validate team/opp
if team_col is None or opp_col is None:
    st.error("Could not detect Team and Opponent columns in schedule transfer CSV. Expected columns like 'Team' and 'Opponent'.")
    st.stop()

# -------------------------
# Build training data (X, y) from daily_df
# -------------------------
# Ensure Points and Opp Points exist in daily_df for targets
points_col = find_col(daily_df, ["Points", "Points "])  # try variant
opp_points_col = find_col(daily_df, ["Opp Points", "Opp Points ", "OppPoints", "Opp_Points"])
if points_col is None or opp_points_col is None:
    st.error("Daily predictor file must contain 'Points' and 'Opp Points' columns for training targets.")
    st.stop()

# numeric features: all numeric columns except target columns
daily_numeric_cols_all = daily_df.select_dtypes(include=[np.number]).columns.tolist()
# remove Points and Opp Points if present
train_feature_cols = [c for c in daily_numeric_cols_all if c not in (points_col, opp_points_col)]

if len(train_feature_cols) == 0:
    st.error("No numeric training features detected in daily predictor file.")
    st.stop()

st.write(f"Using {len(train_feature_cols)} numeric features from Daily predictor for training.")

# Prepare X and y
# convert to numeric and fillna with 0 (you asked predictions without CI; using zeros is safe fallback)
daily_df_numeric = daily_df[train_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
y_points = pd.to_numeric(daily_df[points_col], errors="coerce").fillna(np.nan)
y_opp = pd.to_numeric(daily_df[opp_points_col], errors="coerce").fillna(np.nan)

# Filter rows that have targets
valid_mask = ~y_points.isna() & ~y_opp.isna()
X_train = daily_df_numeric.loc[valid_mask].values
y_train_points = y_points.loc[valid_mask].values
y_train_opp = y_opp.loc[valid_mask].values

if X_train.shape[0] < 10:
    st.warning(f"Only {X_train.shape[0]} usable training rows after filtering missing targets; model may be weak.")

# Optionally include HAN / NonConf flags from daily_df if present (append as features)
# We'll check and append columns if they exist
extra_train_cols = []
daily_han_col = find_col(daily_df, ["HAN", "Han", "Home/Away", "Location", "Loc"])
if daily_han_col:
    # convert HAN to Home flag binary
    han_flags = daily_df[daily_han_col].apply(lambda v: 1 if interpret_han(v) == "Home" else 0).astype(float).fillna(0.0)
    extra_train_cols.append(han_flags)
daily_nonconf_col = find_col(daily_df, ["Non Conference Game", "NonConference", "Non Conf", "NonConf", "Non-Conference"])
if daily_nonconf_col:
    nonconf_flags = daily_df[daily_nonconf_col].apply(lambda v: 1.0 if (str(v).strip().upper() in ("1","TRUE","YES","Y")) else 0.0).fillna(0.0)
    extra_train_cols.append(nonconf_flags)

if extra_train_cols:
    extra_train_mat = np.vstack([c.values for c in extra_train_cols]).T
    X_train = np.hstack([X_train, extra_train_mat[valid_mask]])

# Categorical (coach / conference) encoding if available in daily_df (append ordinal codes)
cat_cols_to_try = ["Coach Name", "Coach", "Coach_Name", "Conference", "Conf", "conference"]
cat_cols = [c for c in cat_cols_to_try if c in daily_df.columns]
cat_encoded_train = None
if cat_cols:
    cat_matrix = daily_df.loc[valid_mask, cat_cols].fillna("NA").astype(str).values
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    try:
        cat_encoded_train = enc.fit_transform(cat_matrix)
        X_train = np.hstack([X_train, cat_encoded_train])
    except Exception:
        cat_encoded_train = None

# Train regressors (Points and Opp Points)
st.write("Training regression models (this retrains on each app run)...")
reg_points = RandomForestRegressor(n_estimators=200, random_state=42)
reg_opp = RandomForestRegressor(n_estimators=200, random_state=42)
try:
    if X_train.shape[0] > 0 and X_train.shape[1] > 0:
        reg_points.fit(X_train, y_train_points)
        reg_opp.fit(X_train, y_train_opp)
    else:
        st.error("Insufficient training data or features to train models.")
        st.stop()
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

st.write("Models trained.")

# -------------------------
# Build prediction features for schedule rows
# -------------------------
# For each feature name in train_feature_cols, attempt to find the same column in schedule_df.
# If not found, try common opponent prefixes/suffixes, otherwise default to 0.
def find_feature_value_in_schedule(row, feat_name):
    # direct match
    if feat_name in row.index:
        v = row.get(feat_name)
        try:
            return float(v) if (pd.notna(v) and v != "") else 0.0
        except Exception:
            return 0.0
    # try opponent or alternate patterns
    patterns = [
        f"Opp {feat_name}", f"Opponent {feat_name}",
        f"OPP_{feat_name}", f"{feat_name}_OPP", f"{feat_name}_opp",
        f"Opp_{feat_name}", f"Opp{feat_name}"
    ]
    for p in patterns:
        if p in row.index:
            v = row.get(p)
            try:
                return float(v) if (pd.notna(v) and v != "") else 0.0
            except Exception:
                return 0.0
    # not found -> 0
    return 0.0

# We'll create a unique-match set to avoid duplicate mirrored rows
seen_match_keys = set()
predictions = []

# Prepare schedule priority sorting values
def conf_priority(conf):
    if pd.isna(conf):
        return 10
    c = str(conf).strip().upper()
    if c in ("SEC",):
        return 2
    if c in ("BIG TEN", "B1G", "BIGTEN", "BIG TEN CONFERENCE"):
        return 3
    if c in ("BIG 12", "BIG12"):
        return 4
    if c in ("ACC",):
        return 5
    return 9

# Determine columns for team-level metadata to display in expander
meta_cols = []
for c in ["Coach Name", "Coach", "Coach_Name", "Conference", "Conf", "Wins", "Losses"]:
    col = find_col(schedule_df, [c])
    if col:
        meta_cols.append(col)

# For rank extraction, candidate rank column patterns
rank_candidates = [c for c in schedule_df.columns if ("RANK" in c.upper()) or c.upper().endswith(" RANK") or c.upper().endswith("_RANK")]

# Build a cleaned schedule DataFrame: deduplicate mirrored rows (same two teams + date)
for idx, r in schedule_df.iterrows():
    t = str(r[team_col]).strip() if pd.notna(r[team_col]) else ""
    o = str(r[opp_col]).strip() if pd.notna(r[opp_col]) else ""
    date = r["__Date_parsed"]
    # create unordered matchup key so Marquette vs Albany and Albany vs Marquette are same
    matchup_key = tuple(sorted([t.lower(), o.lower()])) + (pd.to_datetime(date).date() if not pd.isna(date) else None)
    if matchup_key in seen_match_keys:
        continue
    seen_match_keys.add(matchup_key)

    # priority flags
    is_top25 = False
    if top25_col and top25_col in schedule_df.columns:
        v = r.get(top25_col)
        is_top25 = False
        try:
            if not pd.isna(v) and str(v).strip() not in ("0","", "N", "FALSE", "FALSE "):
                is_top25 = True
        except Exception:
            is_top25 = False

    is_mm = False
    if mm_col and mm_col in schedule_df.columns:
        v = r.get(mm_col)
        try:
            if not pd.isna(v) and str(v).strip() not in ("0","", "N", "FALSE"):
                is_mm = True
        except Exception:
            is_mm = False

    # conference of opponent (for priority)
    opp_conf = r.get("Opponent Conference") if "Opponent Conference" in r.index else r.get("Opp Conference") if "Opp Conference" in r.index else r.get("OpponentConference")
    opp_conf_pr = conf_priority(opp_conf)

    # Build numerical feature vector for this scheduled matchup:
    feat_vals = []
    for feat in train_feature_cols:
        # The schedule file might include the same team-level stats under the same name
        # We'll prefer the team's stat (not an opp stat) assuming column names match
        val = find_feature_value_in_schedule(r, feat)
        feat_vals.append(val)

    # append HAN home flag (prefer schedule HAN column)
    han_flag = 0.0
    if han_col and han_col in r.index:
        hv = interpret_han(r.get(han_col))
        han_flag = 1.0 if hv == "Home" else 0.0
    else:
        # determine by which column equals team or by Location string include 'Madison' etc.
        # We'll try to detect: if schedule has Location and it contains the team's city, but fallback to 0
        han_flag = 0.0

    # append NonConf flag
    nonconf_flag = 0.0
    if nonconf_col and nonconf_col in r.index:
        v = r.get(nonconf_col)
        try:
            nonconf_flag = 1.0 if (not pd.isna(v) and str(v).strip().upper() in ("1","TRUE","YES","Y")) else 0.0
        except Exception:
            nonconf_flag = 0.0
    else:
        # infer via conference columns if available
        tconf = r.get("Conference") if "Conference" in r.index else None
        oconf = r.get("Opponent Conference") if "Opponent Conference" in r.index else None
        try:
            if pd.notna(tconf) and pd.notna(oconf) and str(tconf).strip().upper() != str(oconf).strip().upper():
                nonconf_flag = 1.0
        except Exception:
            nonconf_flag = 0.0

    feat_vals.append(han_flag)
    feat_vals.append(nonconf_flag)

    # categorical columns for this matchup if schedule has them
    cat_vals = []
    for c in ["Coach Name", "Coach", "Coach_Name", "Conference", "Conf"]:
        # pick team-specific or opponent-specific depending on column naming
        if c in r.index:
            cat_vals.append(str(r.get(c)))
        elif f"Team {c}" in r.index:
            cat_vals.append(str(r.get(f"Team {c}")))
        else:
            # try opponent-prefixed to append both team and opponent values
            team_val = r.get(c) if c in r.index else ""
            opp_val = r.get(f"Opponent {c}") if f"Opponent {c}" in r.index else ""
            # fallback: append empty strings to keep counts consistent
            cat_vals.append(str(team_val))
            # if we appended team already, append opponent (so order: team_coach, opp_coach, team_conf, opp_conf, etc.)
            # to keep it simple, append opp too if found:
            if opp_val != "":
                cat_vals.append(str(opp_val))

    # If we have categorical columns and an encoder, transform them (otherwise pad zeros)
    if 'enc' in locals() and len(cat_vals) > 0:
        try:
            cat_enc = enc.transform([cat_vals])
            feat_vec = np.hstack([np.array(feat_vals, dtype=float), cat_enc.ravel()])
        except Exception:
            # unknown categories -> encode as -1
            cat_enc = np.array([[-1] * len(cat_vals)])
            feat_vec = np.hstack([np.array(feat_vals, dtype=float), cat_enc.ravel()])
    else:
        feat_vec = np.array(feat_vals, dtype=float)

    # Align dimensions with X_train
    if feat_vec.size != X_train.shape[1]:
        # create zero-padded vector of training width
        aligned = np.zeros(X_train.shape[1], dtype=float)
        minc = min(feat_vec.size, X_train.shape[1])
        aligned[:minc] = feat_vec[:minc]
        feat_vec = aligned

    # Predict
    try:
        pred_points = float(reg_points.predict(feat_vec.reshape(1, -1))[0])
        pred_opp_points = float(reg_opp.predict(feat_vec.reshape(1, -1))[0])
    except Exception:
        pred_points = 0.0
        pred_opp_points = 0.0

    # Clip to non-negative and round
    pred_points = max(0.0, pred_points)
    pred_opp_points = max(0.0, pred_opp_points)

    # Prepare metadata for expander
    team_meta = {}
    opp_meta = {}
    for meta in ["Coach Name", "Coach", "Conference", "Wins", "Losses"]:
        col_t = find_col(schedule_df, [meta])  # prefer team meta
        col_o = find_col(schedule_df, [f"Opponent {meta}", f"Opp {meta}", f"{meta} (Opp)"])
        # team
        if meta in r.index and pd.notna(r.get(meta)):
            team_meta[meta] = r.get(meta)
        elif col_t and col_t in r.index:
            team_meta[meta] = r.get(col_t)
        else:
            team_meta[meta] = ""
        # opponent
        if f"Opponent {meta}" in r.index and pd.notna(r.get(f"Opponent {meta}")):
            opp_meta[meta] = r.get(f"Opponent {meta}")
        elif col_o and col_o in r.index:
            opp_meta[meta] = r.get(col_o)
        else:
            # fallback: try the symmetric column (if schedule stores both orders)
            opp_meta[meta] = r.get(meta) if pd.notna(r.get(meta)) and str(r.get(team_col)).strip() != str(r.get(opp_col)).strip() else ""

    # Extract top-50 rank categories for both teams from the row
    def extract_top50_for_team(row, team_name):
        # gather any rank columns from the row that appear relevant and have value <= 50
        top50 = []
        for rc in rank_candidates:
            try:
                val = pd.to_numeric(row.get(rc), errors="coerce")
                if pd.notna(val) and val <= 50:
                    top50.append((rc, int(val)))
            except Exception:
                continue
        # This uses the row-level ranks only; if you want global all_stats lookup, expand here.
        return top50

    team_top50 = extract_top50_for_team(r, t)
    opp_top50 = extract_top50_for_team(r, o)

    # Build priority key for sorting:
    # lower tuple sorts earlier: (not is_top25, not is_mm, opp_conf_priority, date)
    priority = (0 if is_top25 else 1, 0 if is_mm else 1, opp_conf_pr, r["__Date_parsed"])

    predictions.append({
        "Date": r["__Date_parsed"],
        "Team": t,
        "Opponent": o,
        "HAN_raw": r.get(han_col) if han_col in r.index else None,
        "HAN": interpret_han(r.get(han_col)) if han_col in r.index else None,
        "IsTop25Opp": is_top25,
        "IsMMOpp": is_mm,
        "Priority": priority,
        "Pred_Points": round(pred_points),
        "Pred_Opp_Points": round(pred_opp_points),
        "Team_Meta": team_meta,
        "Opp_Meta": opp_meta,
        "Team_Top50": team_top50,
        "Opp_Top50": opp_top50
    })

# -------------------------
# Sort predictions by our priority key and date
# -------------------------
pred_df = pd.DataFrame(predictions)
if pred_df.empty:
    st.info("No upcoming games found in schedule file after cleaning.")
    st.stop()

pred_df = pred_df.sort_values(by=["Priority"]).reset_index(drop=True)

# -------------------------
# Display list and expanders with details
# -------------------------
st.subheader("Predicted Upcoming Games")
# Give user ability to filter by conference or show only Top-25 matchups
show_top25_only = st.checkbox("Show only Top-25 opponent games", value=False)
conf_filter = st.multiselect("Filter by opponent conference (optional)", options=sorted(schedule_df.get("Opponent Conference", schedule_df.get("Conference", [])).dropna().unique().tolist()), default=[])

display_rows = []
for _, row in pred_df.iterrows():
    if show_top25_only and not row["IsTop25Opp"]:
        continue
    if conf_filter:
        opp_conf_val = schedule_df.loc[(schedule_df[team_col].astype(str).str.strip() == row["Team"]) & (schedule_df[opp_col].astype(str).str.strip() == row["Opponent"]), "Opponent Conference"]
        opp_conf_val = opp_conf_val.iloc[0] if len(opp_conf_val) > 0 else None
        if not opp_conf_val or opp_conf_val not in conf_filter:
            continue
    display_rows.append(row)

if len(display_rows) == 0:
    st.info("No games match the selected filters.")
else:
    for row in display_rows:
        date_str = row["Date"].strftime("%b %d, %Y") if pd.notna(row["Date"]) else "TBD"
        game_label = f"{row['Team']} vs {row['Opponent']} — {date_str} — Pred: {int(row['Pred_Points'])} - {int(row['Pred_Opp_Points'])}"
        with st.expander(game_label, expanded=False):
            # basic matchup table
            meta_cols_display = ["Coach Name", "Conference", "Wins", "Losses"]
            left = []
            right = []
            for mc in meta_cols_display:
                left.append(mc)
                left.append(row["Team_Meta"].get(mc, ""))
                right.append(row["Opp_Meta"].get(mc, ""))

            # show summary table
            st.markdown("#### Teams")
            col1, col2 = st.columns([1,1])
            with col1:
                st.write(f"**{row['Team']}**")
                for mc in meta_cols_display:
                    st.write(f"{mc}: {row['Team_Meta'].get(mc, '')}")
            with col2:
                st.write(f"**{row['Opponent']}**")
                for mc in meta_cols_display:
                    st.write(f"{mc}: {row['Opp_Meta'].get(mc, '')}")

            st.markdown("---")
            # Top-50 categories
            def format_top50_list(tlist):
                if not tlist:
                    return "None in top 50"
                return pd.DataFrame([{"Category": k, "Rank": v} for k, v in tlist])

            st.markdown("#### Top-50 Rank Categories (from schedule row rank columns)")
            colA, colB = st.columns(2)
            with colA:
                st.write(f"**{row['Team']}**")
                df_a = format_top50_list(row["Team_Top50"])
                st.dataframe(df_a, use_container_width=True)
            with colB:
                st.write(f"**{row['Opponent']}**")
                df_b = format_top50_list(row["Opp_Top50"])
                st.dataframe(df_b, use_container_width=True)

            st.markdown("---")
            st.write("Notes:")
            st.write("- Predictions are per-run model outputs (models retrain on Daily predictor historic games).")
            st.write("- If schedule rows appear duplicated earlier versions of the CSV may include both orders of matchups; this page de-dupes by (team, opponent, date).")
            st.write("- Rank columns are taken from the schedule row; if you want global rank lookups from All_Stats-THE_TABLE.csv we can merge that in as well.")

# -------------------------
# Provide CSV download
# -------------------------
out_for_csv = pred_df[["Date", "Team", "Opponent", "HAN_raw", "Pred_Points", "Pred_Opp_Points", "IsTop25Opp", "IsMMOpp"]].copy()
out_for_csv["Date"] = out_for_csv["Date"].dt.strftime("%Y-%m-%d")
csv = out_for_csv.to_csv(index=False)
st.download_button("Download predictions CSV", csv, "todays_games_predictions.csv", "text/csv")
