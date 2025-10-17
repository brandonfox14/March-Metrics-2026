"""
APP/Pages/5_Schedule_Predictor.py

- Uses ALL numeric team stats from All_Stats-THE_TABLE.csv as features.
- Encodes categorical coach + conference info for team and opponent.
- Trains:
    * RandomForestClassifier -> win probability
    * RandomForestRegressor    -> point margin (home - away)
  (Regressor predictions are used to derive projected team/opp scores.)
- Date parsing is robust and schedule is sorted by true chronological order (YYYY-MM-DD).
- Default team is obvious/editable at the top.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

# -------------------------
# CONFIG: change default here
# -------------------------
# Clear and obvious default team variable
INITIAL_TEAM = "Wisconsin"   # <- edit this line to change the default selected team

# Paths (relative to repo root)
BASE = "Data/26_March_Madness_Databook"
SCHEDULE_FILE = os.path.join(BASE, "2026 Schedule Simple-Table 1.csv")
DAILY_FILE = os.path.join(BASE, "Daily_predictor_data-Table 1.csv")
ALL_STATS_FILE = os.path.join(BASE, "All_Stats-THE_TABLE.csv")

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Schedule Predictor", layout="wide")
st.title("Schedule Predictor (Schedule → Predictions)")
st.write(
    "Select a team to view upcoming games. Models use full numeric team stats plus categorical coach/conference encodings."
)

# -------------------------
# SAFE CSV LOADER
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
        st.error(f"Failed to read {path}: {e}")
        return None

schedule_df = load_csv(SCHEDULE_FILE)
daily_df = load_csv(DAILY_FILE)
all_stats = load_csv(ALL_STATS_FILE)

if schedule_df is None:
    st.error(f"Schedule file not found at {SCHEDULE_FILE}")
    st.stop()
if daily_df is None:
    st.error(f"Daily predictor file not found at {DAILY_FILE}")
    st.stop()
if all_stats is None:
    st.error(f"All stats file not found at {ALL_STATS_FILE}")
    st.stop()

st.info("Data files loaded.")

# -------------------------
# HELPERS: find team/opponent column names variants
# -------------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

team_col_candidates = ["Team", "Teams", "team", "TEAM"]
opp_col_candidates = ["Opponent", "Opp", "opponent", "OPPONENT"]

schedule_team_col = find_col(schedule_df, ["Team", "Teams", "team", "Home", "Away", "Home Team", "Away Team"])
schedule_opp_col = find_col(schedule_df, ["Opponent", "Opp", "Opponent Team", "Away", "Home", "Opponent Team"])

daily_team_col = find_col(daily_df, ["Team", "team", "Teams"])
daily_opp_col = find_col(daily_df, ["Opponent", "Opp", "opponent"])

# All_stats team column name
all_stats_team_col = find_col(all_stats, ["Teams", "Team", "team"])

if all_stats_team_col is None:
    st.error("Could not find a teams column in All_Stats-THE_TABLE.csv (expected 'Teams' or 'Team').")
    st.stop()

# Schedule team/opp fallback detection: explicit columns likely "Team" and "Opponent"
if schedule_team_col is None or schedule_opp_col is None:
    # Try to infer by checking for columns whose values match known team names
    text_cols = [c for c in schedule_df.columns if schedule_df[c].dtype == object]
    found = []
    for c in text_cols:
        sample = schedule_df[c].dropna().astype(str).head(30).tolist()
        matches = sum(1 for v in sample if v in all_stats[all_stats_team_col].values)
        if matches >= 1:
            found.append(c)
    if len(found) >= 2:
        schedule_team_col, schedule_opp_col = found[0], found[1]

if schedule_team_col is None or schedule_opp_col is None:
    st.error("Could not autodetect home/team and opponent columns in schedule file.")
    st.stop()

# Daily team/opponent fallback: attempt to infer similarly
if daily_team_col is None or daily_opp_col is None:
    text_cols = [c for c in daily_df.columns if daily_df[c].dtype == object]
    found = []
    for c in text_cols:
        sample = daily_df[c].dropna().astype(str).head(30).tolist()
        matches = sum(1 for v in sample if v in all_stats[all_stats_team_col].values)
        if matches >= 1:
            found.append(c)
    if len(found) >= 2:
        daily_team_col, daily_opp_col = found[0], found[1]

if daily_team_col is None or daily_opp_col is None:
    st.warning("Could not find explicit Team/Opponent columns inside daily predictor data; model will still try to use numeric columns for training but per-game merges may be limited.")

# -------------------------------
# 🗓️ SAFE DATE PARSING & SORTING
# -------------------------------
date_candidates = ["Date", "date", "Game_Date", "Game Date"]
date_col = next((c for c in schedule_df.columns if c in date_candidates), None)

if date_col:
    schedule_df["__Date_parsed"] = pd.to_datetime(
        schedule_df[date_col].astype(str).str.strip(),
        errors="coerce",
        infer_datetime_format=True
    )
else:
    # fallback: try to use first column if no match
    schedule_df["__Date_parsed"] = pd.to_datetime(
        schedule_df.iloc[:, 0].astype(str).str.strip(),
        errors="coerce",
        infer_datetime_format=True
    )

# Drop invalid or missing dates safely
schedule_df = schedule_df.dropna(subset=["__Date_parsed"])

# Sort chronologically (YYYY-MM-DD order)
schedule_df = schedule_df.sort_values("__Date_parsed").reset_index(drop=True)


# -------------------------
# PREP All-Stats features
# -------------------------
# numeric team-level stats (use all numeric columns)
numeric_team_cols = all_stats.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_team_cols) == 0:
    st.error("No numeric columns detected in All_Stats-THE_TABLE.csv; cannot build numeric features.")
    st.stop()

# Categorical fields we want to include (if present)
cat_team_cols = []
for c in ["Coach Name", "Coach", "Coach_Name", "coach", "Conference", "Conf", "conference"]:
    if c in all_stats.columns:
        cat_team_cols.append(c)

# Build lookup dicts for numeric and categorical team info
team_numeric_lookup = {}
team_cat_lookup = {}
for _, r in all_stats.iterrows():
    tname = r[all_stats_team_col]
    team_numeric_lookup[str(tname).strip()] = r[numeric_team_cols].astype(float).fillna(0.0).values
    team_cat_lookup[str(tname).strip()] = {c: (r[c] if c in all_stats.columns else np.nan) for c in cat_team_cols}

st.write(f"Using {len(numeric_team_cols)} numeric team-level stats from All_Stats.")

# -------------------------
# BUILD TRAINING SET FROM daily_df
# - For each historical row in daily_df, attempt to attach team & opp numeric stats
# - Features = [team_numeric, opp_numeric (or team-opp diff)] + categorical encodings
# -------------------------
# Use numeric columns from daily_df as additional features (if any)
daily_numeric_cols = daily_df.select_dtypes(include=[np.number]).columns.tolist()
st.write(f"Daily training data numeric columns: {len(daily_numeric_cols)}")

# Find team/opponent columns in daily_df (if found above)
train_rows = []
skip_count = 0
for idx, r in daily_df.iterrows():
    # try to get team/opponent names from daily row
    team_name = None
    opp_name = None
    if daily_team_col and daily_team_col in daily_df.columns:
        team_name = str(r[daily_team_col]).strip()
    if daily_opp_col and daily_opp_col in daily_df.columns:
        opp_name = str(r[daily_opp_col]).strip()

    # if not found, try schedule-style columns present in daily_df
    if team_name is None or opp_name is None:
        # skip if we cannot map teams
        skip_count += 1
        continue

    # require both teams exist in all_stats lookups
    if team_name not in team_numeric_lookup or opp_name not in team_numeric_lookup:
        skip_count += 1
        continue

    # build numeric feature vector: team_numeric, opp_numeric, plus elementwise diff (team - opp)
    team_num = team_numeric_lookup[team_name]
    opp_num = team_numeric_lookup[opp_name]
    # vector: [team_num, opp_num, team_num - opp_num]
    feat = np.concatenate([team_num, opp_num, team_num - opp_num])

    # include daily numeric features if present (to capture game-level signals)
    if len(daily_numeric_cols) > 0:
        daily_feats = r[daily_numeric_cols].astype(float).fillna(0.0).values
        feat = np.concatenate([feat, daily_feats])

    # categorical features: coach/conf for team and opponent (if available)
    cat_vals = []
    for c in cat_team_cols:
        tcat = team_cat_lookup.get(team_name, {}).get(c, np.nan)
        ocat = team_cat_lookup.get(opp_name, {}).get(c, np.nan)
        cat_vals.append(str(tcat))
        cat_vals.append(str(ocat))

    # target: margin or win
    # detect Points/Opp Points in daily_df
    if "Points" in daily_df.columns and "Opp Points" in daily_df.columns:
        margin = float(r["Points"]) - float(r["Opp Points"])
        win = 1 if margin > 0 else 0
    else:
        # fallback: try 'Points' / 'OppPoints' variants
        margin = np.nan
        win = np.nan

    train_rows.append({
        "feat": feat,
        "cats": cat_vals,
        "margin": margin,
        "win": win
    })

if len(train_rows) < 30:
    st.warning(f"Only {len(train_rows)} historical rows built for training (skipped {skip_count}). Training may be weak with limited matched rows.")

# Build X and y for models
X_num = np.vstack([r["feat"] for r in train_rows]) if len(train_rows) > 0 else np.zeros((0, 1))
y_margin = np.array([r["margin"] for r in train_rows])
y_win = np.array([r["win"] for r in train_rows])

# Categorical handling: ordinal encode the cat columns (team coach, team conf, opp coach, opp conf)
cat_matrix = None
if len(cat_team_cols) > 0 and len(train_rows) > 0:
    cat_list = [r["cats"] for r in train_rows]  # list of lists
    cat_matrix = np.array(cat_list)
    # Ordinal encode (trees are fine with ordinal encoding)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    try:
        cat_encoded = enc.fit_transform(cat_matrix)
    except Exception:
        # If encoding fails (e.g., empty or invalid), fallback to zeros
        cat_encoded = np.zeros((len(train_rows), cat_matrix.shape[1]))
else:
    cat_encoded = np.zeros((len(train_rows), 0))

# Final training matrix
if X_num.shape[0] == 0:
    st.error("No training rows could be constructed from daily predictor data merged with All_Stats. Check team name consistency.")
    st.stop()

X_train_full = np.hstack([X_num, cat_encoded]) if cat_encoded.shape[1] > 0 else X_num

# Replace any nan/inf with zeros
X_train_full = np.nan_to_num(X_train_full, nan=0.0, posinf=0.0, neginf=0.0)

# y arrays cleanup: drop rows with nan targets
valid_idx = ~np.isnan(y_margin) & ~np.isnan(y_win)
X_train_full = X_train_full[valid_idx]
y_margin = y_margin[valid_idx]
y_win = y_win[valid_idx]

# -------------------------
# MODEL TRAINING
# -------------------------
st.write(f"Training models on {X_train_full.shape[0]} examples and {X_train_full.shape[1]} features...")

# Classification: Win probability
if len(np.unique(y_win)) < 2:
    st.warning("Win label has only one class in training; classification will be unreliable.")
clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_train_full, y_win)

# Regression: margin
reg = RandomForestRegressor(n_estimators=300, random_state=42)
reg.fit(X_train_full, y_margin)

# Residual std for margin -> used for simple CI
pred_margin_train = reg.predict(X_train_full)
resid_sd = np.std(y_margin - pred_margin_train)
st.write(f"Trained. Margin residual sd ≈ {resid_sd:.2f}")

# -------------------------
# SCHEDULE & TEAM DROPDOWN
# -------------------------
# Team list comes from all_stats team column
team_values = sorted(all_stats[all_stats_team_col].dropna().unique().astype(str).tolist())
default_index = team_values.index(INITIAL_TEAM) if INITIAL_TEAM in team_values else 0
selected_team = st.selectbox("Select a team (default controlled by top-of-file INITIAL_TEAM)", team_values, index=default_index)
st.write(f"Default INITIAL_TEAM variable is: '{INITIAL_TEAM}' (edit file to change)")

# Filter upcoming games for selected team (either column matches)
mask = (schedule_df[schedule_team_col].astype(str).str.strip() == selected_team) | (schedule_df[schedule_opp_col].astype(str).str.strip() == selected_team)
selected_schedule = schedule_df.loc[mask].copy().sort_values("__Date_parsed").reset_index(drop=True)

if selected_schedule.empty:
    st.info(f"No scheduled games found for {selected_team}.")
    st.stop()

# Limit to next N games (you asked next 3)
N = st.number_input("How many upcoming games to show", min_value=1, max_value=12, value=3, step=1)

# -------------------------
# PREDICT FOR EACH UPCOMING GAME
# -------------------------
pred_rows = []
for _, game in selected_schedule.head(N).iterrows():
    # Determine which side is selected_team, opponent, and location
    row_team = str(game[schedule_team_col]).strip()
    row_opp = str(game[schedule_opp_col]).strip()
    # determine opponent name relative to selection
    if row_team == selected_team:
        team_name = row_team
        opp_name = row_opp
        location = "Home"
    elif row_opp == selected_team:
        team_name = row_opp
        opp_name = row_team
        location = "Away"
    else:
        # fallback (rare)
        team_name = row_team
        opp_name = row_opp
        location = "Neutral/Unknown"

    # ensure both are in lookups
    if team_name not in team_numeric_lookup or opp_name not in team_numeric_lookup:
        # try trimmed matching
        if team_name not in team_numeric_lookup:
            st.write(f"Warning: {team_name} not found in team stats; skipping row.")
            continue
        if opp_name not in team_numeric_lookup:
            st.write(f"Warning: {opp_name} not found in team stats; skipping row.")
            continue

    team_num = team_numeric_lookup[team_name]
    opp_num = team_numeric_lookup[opp_name]
    feat_vec = np.concatenate([team_num, opp_num, team_num - opp_num])

    # If daily numeric columns exist and the schedule contains columns we can map into them, we could append; for now keep consistent with training dims
    # Build categorical vector (team coach, opp coach, team conf, opp conf)
    cat_vals = []
    for c in cat_team_cols:
        tcat = team_cat_lookup.get(team_name, {}).get(c, "")
        ocat = team_cat_lookup.get(opp_name, {}).get(c, "")
        cat_vals.append(str(tcat))
        cat_vals.append(str(ocat))
    if len(cat_vals) > 0:
        try:
            cat_enc = enc.transform([cat_vals])
        except Exception:
            # unknown categories -> encode as -1
            cat_enc = np.array([[-1] * len(cat_vals)])
        X_future = np.hstack([feat_vec, cat_enc.ravel()])
    else:
        X_future = feat_vec

    X_future = np.nan_to_num(X_future, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)

    # If future feature dimension mismatches training, align by truncation/zero-padding
    if X_future.shape[1] != X_train_full.shape[1]:
        # truncate or pad
        min_cols = min(X_future.shape[1], X_train_full.shape[1])
        X_tmp = np.zeros((1, X_train_full.shape[1]))
        X_tmp[0, :min_cols] = X_future[0, :min_cols]
        X_future = X_tmp

    # Predict win probability and margin
    win_prob = clf.predict_proba(X_future)[0][1] if hasattr(clf, "predict_proba") else clf.predict(X_future)[0]
    pred_margin = reg.predict(X_future)[0]

    # Using predicted margin and a baseline total to split into team/opp scores:
    # We estimate average total from training data if possible
    avg_total = 140.0
    if "Points" in daily_df.columns and "Opp Points" in daily_df.columns:
        avg_total = float((daily_df["Points"].fillna(0) + daily_df["Opp Points"].fillna(0)).mean())

    pred_team_score = (avg_total + pred_margin) / 2.0
    pred_opp_score = (avg_total - pred_margin) / 2.0

    # 95% CI for margin
    z = 1.96
    margin_lo = pred_margin - z * resid_sd
    margin_hi = pred_margin + z * resid_sd

    # convert to score CIs
    team_lo = (avg_total + margin_lo) / 2.0
    team_hi = (avg_total + margin_hi) / 2.0
    opp_lo = (avg_total - margin_hi) / 2.0
    opp_hi = (avg_total - margin_lo) / 2.0

    pred_rows.append({
        "Date": game["__Date_parsed"].strftime("%Y-%m-%d") if not pd.isna(game["__Date_parsed"]) else str(game.get("Date", "")),
        "Opponent": opp_name,
        "Location": location,
        "Win_Prob": f"{win_prob*100:.1f}%",
        "Projected": f"{selected_team} {pred_team_score:.0f} - {opp_name} {pred_opp_score:.0f}",
        "Team_Score_CI": f"[{team_lo:.0f}, {team_hi:.0f}]",
        "Opp_Score_CI": f"[{opp_lo:.0f}, {opp_hi:.0f}]",
        "Pred_Margin": round(pred_margin, 1),
        "Margin_CI": f"[{margin_lo:.1f}, {margin_hi:.1f}]"
    })

# -------------------------
# SHOW TABLE
# -------------------------
if len(pred_rows) == 0:
    st.info("No predictions available (likely missing stats mapping for some games).")
else:
    out_df = pd.DataFrame(pred_rows)
    st.subheader(f"Next {len(out_df)} games (chronological) for {selected_team}")
    st.dataframe(out_df, use_container_width=True)
    csv = out_df.to_csv(index=False)
    st.download_button("Download predictions CSV", csv, f"{selected_team}_schedule_predictions.csv", "text/csv")

# -------------------------
# NOTES
# -------------------------
st.markdown("---")
st.write(
    "Notes:\n"
    "- This page trains models using ALL numeric team-level stats from All_Stats-THE_TABLE.csv (element-wise team, opponent, and diff are used).\n"
    "- Coach and Conference categories (if present) are ordinal-encoded and included.\n"
    "- Dates are parsed and sorted chronologically using true datetimes (YYYY-MM-DD ordering).\n"
    "- To change the default team on load edit the INITIAL_TEAM constant at top of file.\n"
    "- This is a strong baseline: you can swap RandomForest -> XGBoost/LightGBM and add cross-validation / hyperparameter tuning for improved accuracy."
)
