"""
Pages/5_Schedule_Predictor.py

Schedule Predictor:
- Trains a model on historical 'daily_predictor_data' to predict point margin (home - away)
- Uses team-level stats from All_Stats-THE_TABLE.csv to construct features for future games
- Predicts expected final scores and 95% confidence intervals for each future matchup

HOW TO SET THE INITIAL TEAM:
- Edit the constant INITIAL_TEAM below (it's obvious and isolated).
- You can also override it from the UI.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# -------------------------
# ADJUSTABLE INITIAL TEAM
# -------------------------
# Change this constant to set the initial team selected when the page loads.
# Example: INITIAL_TEAM = "Wisconsin"
INITIAL_TEAM = "Wisconsin"   # <-- change this line if you want a different default

# -------------------------
# CONFIG / FILE PATHS
# -------------------------
BASE = "Data/26_March_Madness_Databook"
SCHEDULE_FILE = os.path.join(BASE, "2026 Schedule Simple-Table 1.csv")
DAILY_FILE = os.path.join(BASE, "Daily_predictor_data-Table 1.csv")
ALL_STATS_FILE = os.path.join(BASE, "All_Stats-THE_TABLE.csv")

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Schedule Predictor", layout="wide")
st.title("Schedule Predictor")

st.write(
    """
Use historical game-level training data and current team statistics to forecast upcoming matchups.
This tool predicts **point margin** (home − away) and produces an expected final score with a 95% confidence interval.
"""
)

# -------------------------
# Helper: safe CSV loader
# -------------------------
@st.cache_data
def load_csv_safe(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, encoding="latin1")
    except Exception as e:
        st.warning(f"Error reading {path}: {e}")
        return None

# -------------------------
# Load data
# -------------------------
schedule_df = load_csv_safe(SCHEDULE_FILE)
daily_df = load_csv_safe(DAILY_FILE)
all_stats = load_csv_safe(ALL_STATS_FILE)

if schedule_df is None:
    st.error(f"Schedule file not found at: {SCHEDULE_FILE}")
    st.stop()
if daily_df is None:
    st.error(f"Daily predictor (training) file not found at: {DAILY_FILE}")
    st.stop()
if all_stats is None:
    st.error(f"All stats file not found at: {ALL_STATS_FILE}")
    st.stop()

st.success("Data files loaded successfully.")

# -------------------------
# Inspect / normalize columns
# -------------------------
# standardize column names to strip spaces
schedule_df.columns = schedule_df.columns.str.strip()
daily_df.columns = daily_df.columns.str.strip()
all_stats.columns = all_stats.columns.str.strip()

st.markdown("### Quick data diagnostics")
col1, col2, col3 = st.columns(3)
with col1:
    st.write("Schedule rows:", len(schedule_df))
with col2:
    st.write("Training rows (daily):", len(daily_df))
with col3:
    st.write("Known teams in All_Stats:", all_stats["Teams"].nunique() if "Teams" in all_stats.columns else "Teams col missing")

st.markdown("---")

# -------------------------
# Detect target (margin) in training data
# -------------------------
# We try a few common column names. If home/away scores exist, compute margin.
def detect_margin_column(df):
    possible_margin_names = ["margin", "point_margin", "point diff", "home_minus_away", "Margin"]
    for n in possible_margin_names:
        if n in df.columns:
            return n
    # else look for home/away score combos
    home_candidates = ["home_score", "home_pts", "HomeScore", "Home_PTS", "Home_PTS"]
    away_candidates = ["away_score", "away_pts", "AwayScore", "Away_PTS", "Away_PTS"]
    h = next((c for c in home_candidates if c in df.columns), None)
    a = next((c for c in away_candidates if c in df.columns), None)
    if h and a:
        df["_TRAIN_HOME_SCORE"] = pd.to_numeric(df[h], errors="coerce")
        df["_TRAIN_AWAY_SCORE"] = pd.to_numeric(df[a], errors="coerce")
        df["_TRAIN_MARGIN"] = df["_TRAIN_HOME_SCORE"] - df["_TRAIN_AWAY_SCORE"]
        return "_TRAIN_MARGIN"
    return None

margin_col = detect_margin_column(daily_df)
if margin_col is None:
    st.error("Could not detect a margin (target) column in daily predictor data. Please include home/away score columns or a margin column.")
    st.stop()

st.write("Detected margin column:", margin_col)

# -------------------------
# Feature preparation for training
# -------------------------
# Strategy:
# - Use numeric columns in daily_df (excluding identifiers, dates, team names) as training features.
# - If daily_df already contains engineered features (home_* and away_*), they'll be used automatically.
def prepare_training(df, target_col):
    df = df.copy()
    # drop obviously non-feature columns
    exclude = set([target_col, "date", "game_id", "GameID", "home_team", "away_team", "home", "away", "Home", "Away"])
    # numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    X = df[numeric_cols].astype(float).fillna(0.0)
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)
    return X, y, numeric_cols

X_train, y_train, train_feature_cols = prepare_training(daily_df, margin_col)
st.write(f"Using {len(train_feature_cols)} numeric features from training data.")

# -------------------------
# Model training
# -------------------------
# Use scikit-learn if available; otherwise fall back to linear regression from numpy
use_sklearn = True
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
except Exception:
    use_sklearn = False

if not use_sklearn:
    st.warning("scikit-learn not available in environment. Install scikit-learn for a better model: `pip install scikit-learn`.")
else:
    st.write("Training GradientBoostingRegressor on margin (home - away)...")
    # Basic split and train
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X_tr, y_tr)
    val_pred = model.predict(X_val)
    val_resid = y_val - val_pred
    resid_sd = np.std(val_resid)
    st.write(f"Validation RMSE: {np.sqrt(np.mean((val_resid)**2)):.3f}, residual sd ~ {resid_sd:.3f}")

# -------------------------
# Build features for future schedule
# -------------------------
# Assumptions:
# - schedule_df includes columns for home team and away team (common names: Home, Away, HOME_TEAM, AWAY_TEAM)
# - all_stats has a 'Teams' column matching schedule team names
team_cols_candidates = ["Home", "Away", "HOME_TEAM", "AWAY_TEAM", "home_team", "away_team", "Team1", "Team2"]
home_col = next((c for c in team_cols_candidates if c in schedule_df.columns), None)
away_col = next((c for c in team_cols_candidates if c in schedule_df.columns and c != home_col), None)

# try to find sensible column names
if home_col is None or away_col is None:
    # try a heuristic: look for two columns that contain team names by sampling values
    text_cols = [c for c in schedule_df.columns if schedule_df[c].dtype == object]
    found = []
    for c in text_cols:
        sample_vals = schedule_df[c].dropna().astype(str).head(20).tolist()
        # check if sample values exist in all_stats Teams
        if "Teams" in all_stats.columns:
            matches = sum([1 for v in sample_vals if v in all_stats["Teams"].values])
            if matches >= 1:
                found.append(c)
    if len(found) >= 2:
        home_col, away_col = found[0], found[1]

if home_col is None or away_col is None:
    st.error("Could not automatically detect home/away team columns in schedule. Please confirm the schedule file contains clear team name columns.")
    st.stop()

st.write(f"Using schedule team columns: home='{home_col}', away='{away_col}'")

# Build team stat features: find numeric columns in all_stats to represent teams
numeric_team_cols = [c for c in all_stats.columns if pd.api.types.is_numeric_dtype(all_stats[c]) and c not in ("Wins","Losses")]
# ensure Teams column exists
if "Teams" not in all_stats.columns:
    st.error("'Teams' column missing from All_Stats-THE_TABLE.csv")
    st.stop()

# Create a lookup of team -> numeric stat vector (fillna with zeros)
team_stats_lookup = {}
for _, row in all_stats.iterrows():
    team = row["Teams"]
    stats = row[numeric_team_cols].astype(float).fillna(0.0).values
    team_stats_lookup[team] = stats

# Prepare schedule features (home - away numeric stat differences)
future_rows = []
for _, r in schedule_df.iterrows():
    home = str(r[home_col]).strip()
    away = str(r[away_col]).strip()
    # skip if teams not found
    if home not in team_stats_lookup or away not in team_stats_lookup:
        continue
    home_stats = team_stats_lookup[home]
    away_stats = team_stats_lookup[away]
    feat = home_stats - away_stats
    future_rows.append({
        "date": r.get("Date", r.get("date", np.nan)),
        "home": home,
        "away": away,
        "features": feat
    })

if len(future_rows) == 0:
    st.error("No future schedule rows could be constructed with matching team names in All_Stats.")
    st.stop()

# Convert to DataFrame
feature_matrix = np.vstack([r["features"] for r in future_rows])
future_df = pd.DataFrame(feature_matrix, columns=[f"tstat_{c}" for c in numeric_team_cols])
future_df["home"] = [r["home"] for r in future_rows]
future_df["away"] = [r["away"] for r in future_rows]
future_df["date"] = [r["date"] for r in future_rows]

# -------------------------
# Align future features with training features (if possible)
# -------------------------
# If training used explicit columns, try to map. Otherwise, train model uses numeric columns from daily_df.
# If number of features mismatch, we'll attempt to use only overlapping features.
if use_sklearn:
    # If X_train columns length equals future feature length -> use directly
    if feature_matrix.shape[1] == X_train.shape[1]:
        X_future = future_df.iloc[:, :feature_matrix.shape[1]]
    else:
        # Try to reduce/expand: use PCA or simple alignment; for now, try to match by position if counts are close.
        min_cols = min(feature_matrix.shape[1], X_train.shape[1])
        X_future = future_df.iloc[:, :min_cols]
        # Also truncate train columns similarly for prediction
        # We'll build a reduced model on those min_cols
        X_reduced = X_train.iloc[:, :min_cols]
        # retrain on reduced features for better alignment
        model = GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)
        model.fit(X_reduced, y_train)
        # reduce X_future to same
        X_future = X_future.copy()
else:
    st.warning("Modeling skipped because scikit-learn is not available; returning simple baseline predictions.")
    # baseline: predict zero margin and use team average total points for scores
    future_df["pred_margin"] = 0.0
    future_df["pred_total"] = 140.0  # arbitrary baseline
    # compute naive scores
    future_df["pred_home"] = (future_df["pred_total"] + future_df["pred_margin"]) / 2
    future_df["pred_away"] = (future_df["pred_total"] - future_df["pred_margin"]) / 2
    st.write(future_df[["date", "home", "away", "pred_home", "pred_away"]].head(20))
    st.stop()

# If model exists (trained earlier)
if use_sklearn:
    # if min_cols branch happened, ensure X_future shape aligns with model input
    try:
        # If we retrained a reduced model earlier, model is available
        X_for_pred = X_future.values.astype(float)
        preds_margin = model.predict(X_for_pred)
    except Exception as e:
        st.warning(f"Failed to predict using model directly: {e}")
        # As fallback, predict zeros
        preds_margin = np.zeros(X_future.shape[0])

    # Estimate residual std for CI: use val_resid (if available) else training residuals
    try:
        resid_sd = float(resid_sd)
    except Exception:
        resid_sd = np.std(y_train - model.predict(X_train)) if X_train.shape[0] > 0 else 10.0

    # Predict total points baseline: use average of (home_score + away_score) from training if available
    total_cols = [c for c in daily_df.columns if "total" in c.lower() or "tot" in c.lower()]
    if total_cols:
        avg_total = float(daily_df[total_cols[0]].mean())
    else:
        # try sum of detected home/away if present
        if "_TRAIN_HOME_SCORE" in daily_df.columns and "_TRAIN_AWAY_SCORE" in daily_df.columns:
            avg_total = float((daily_df["_TRAIN_HOME_SCORE"] + daily_df["_TRAIN_AWAY_SCORE"]).mean())
        else:
            avg_total = 140.0  # fallback baseline

    # Build predictions dataframe
    preds = []
    z = 1.96  # 95% CI
    for i, row in future_df.iterrows():
        m_hat = float(preds_margin[i])
        # CI for margin
        margin_lower = m_hat - z * resid_sd
        margin_upper = m_hat + z * resid_sd
        # infer predicted totals (use avg_total)
        tot_hat = avg_total
        pred_home = (tot_hat + m_hat) / 2.0
        pred_away = (tot_hat - m_hat) / 2.0
        # CI converted to scores
        home_lower = (tot_hat + margin_lower) / 2.0
        home_upper = (tot_hat + margin_upper) / 2.0
        away_lower = (tot_hat - margin_upper) / 2.0  # note swap since margin_upper reduces away
        away_upper = (tot_hat - margin_lower) / 2.0
        preds.append({
            "date": row["date"],
            "home": row["home"],
            "away": row["away"],
            "pred_margin": m_hat,
            "margin_lo": margin_lower,
            "margin_hi": margin_upper,
            "pred_total": tot_hat,
            "pred_home": pred_home,
            "pred_away": pred_away,
            "home_lo": home_lower,
            "home_hi": home_upper,
            "away_lo": away_lower,
            "away_hi": away_upper
        })

    preds_df = pd.DataFrame(preds)
    # sort by date if possible
    try:
        preds_df["date_parsed"] = pd.to_datetime(preds_df["date"])
        preds_df = preds_df.sort_values("date_parsed").drop(columns=["date_parsed"])
    except Exception:
        pass

    # -------------------------
    # UI: allow overriding initial team (UI control) while leaving code constant
    # -------------------------
    st.markdown("### Controls")
    init_team_ui = st.text_input("Initial team (code default is the constant INITIAL_TEAM)", value=INITIAL_TEAM)
    st.write(f"Code default INITIAL_TEAM = '{INITIAL_TEAM}' (change the constant at top of file to alter the default)")

    # filter by team if desired
    team_filter = st.selectbox("Filter predictions for a specific team (home or away) or choose 'All'", options=["All"] + sorted(list(preds_df["home"].unique())))
    if team_filter != "All":
        preds_show = preds_df[(preds_df["home"] == team_filter) | (preds_df["away"] == team_filter)].copy()
    else:
        preds_show = preds_df.copy()

    # Show results
    st.markdown("---")
    st.subheader("Predicted upcoming games")
    display_cols = ["date", "home", "away", "pred_home", "home_lo", "home_hi", "pred_away", "away_lo", "away_hi", "pred_margin", "margin_lo", "margin_hi"]
    # format floats
    for col in ["pred_home", "home_lo", "home_hi", "pred_away", "away_lo", "away_hi", "pred_margin", "margin_lo", "margin_hi"]:
        preds_show[col] = preds_show[col].round(1)

    st.dataframe(preds_show[display_cols].reset_index(drop=True), use_container_width=True)

    # Provide download button CSV
    csv = preds_show[display_cols].to_csv(index=False)
    st.download_button("Download predictions CSV", csv, "schedule_predictions.csv", "text/csv")

    st.markdown("---")
    st.write("Notes and assumptions:")
    st.markdown(
        """
        - The model predicts **point margin** (home - away) using numeric features found in the training file.
        - Future game features are constructed using team-level numeric stats from `All_Stats-THE_TABLE.csv` by taking **home - away** for each numeric stat.
        - Predicted final scores are estimated by combining the predicted margin with an average total points baseline from training data.
        - Confidence intervals are approximated assuming residuals are roughly normal; for more accurate predictive intervals consider bootstrapping or quantile regressors.
        - If team names in the schedule don't match the `Teams` values in All_Stats exactly, those games are skipped — ensure naming consistency.
        - To change the default initial team, edit the `INITIAL_TEAM` constant at the top of this file.
        """
    )
