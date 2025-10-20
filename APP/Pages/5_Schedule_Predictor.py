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
# HELPERS: find team/opponent/other column name variants
# -------------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# candidate names
team_candidates = ["Team", "Teams", "team", "HOME_TEAM", "Home", "Home Team"]
opp_candidates = ["Opponent", "Opp", "OPPONENT", "Away", "Away Team"]
date_candidates = ["Date", "date", "Game_Date", "Game Date"]
han_candidates = ["HAN", "Han", "Home/Away", "Location", "Loc", "HomeAway", "HOME/ AWAY", "HOME/AWAY"]
nonconf_candidates = ["Non Conference Game", "NonConference", "Non Conf", "NonConf", "Non Conference", "Non-Conference"]
coach_candidates = ["Coach Name", "Coach", "Coach_Name", "coach"]
conf_candidates = ["Conference", "Conf", "conference"]

# detect columns
schedule_team_col = find_col(schedule_df, team_candidates)
schedule_opp_col = find_col(schedule_df, opp_candidates)
date_col = find_col(schedule_df, date_candidates)
han_col = find_col(schedule_df, han_candidates)
nonconf_col = find_col(schedule_df, nonconf_candidates)

daily_team_col = find_col(daily_df, team_candidates)
daily_opp_col = find_col(daily_df, opp_candidates)
daily_han_col = find_col(daily_df, han_candidates)
daily_nonconf_col = find_col(daily_df, nonconf_candidates)

all_stats_team_col = find_col(all_stats, ["Teams", "Team", "team"])
all_stats_coach_col = find_col(all_stats, coach_candidates)
all_stats_conf_col = find_col(all_stats, conf_candidates)

# if team columns not found in schedule or all_stats, attempt heuristics
if all_stats_team_col is None:
    st.error("Could not find a teams column in All_Stats-THE_TABLE.csv (expected 'Teams' or 'Team').")
    st.stop()

if schedule_team_col is None or schedule_opp_col is None:
    # try to infer which two text columns contain team names
    text_cols = [c for c in schedule_df.columns if schedule_df[c].dtype == object]
    found = []
    for c in text_cols:
        sample = schedule_df[c].dropna().astype(str).head(40).tolist()
        matches = sum(1 for v in sample if v in all_stats[all_stats_team_col].values)
        if matches >= 1:
            found.append(c)
    if len(found) >= 2:
        schedule_team_col, schedule_opp_col = found[0], found[1]

if schedule_team_col is None or schedule_opp_col is None:
    st.error("Could not autodetect home/team and opponent columns in schedule file. Please ensure one column lists the team and another the opponent.")
    st.stop()

# attempt similar for daily_df but warn rather than stop
if daily_team_col is None or daily_opp_col is None:
    text_cols = [c for c in daily_df.columns if daily_df[c].dtype == object]
    found = []
    for c in text_cols:
        sample = daily_df[c].dropna().astype(str).head(40).tolist()
        matches = sum(1 for v in sample if v in all_stats[all_stats_team_col].values)
        if matches >= 1:
            found.append(c)
    if len(found) >= 2:
        daily_team_col, daily_opp_col = found[0], found[1]

if daily_team_col is None or daily_opp_col is None:
    st.warning("Could not find Team/Opponent columns inside daily predictor data; training will still use numeric stats but some per-game merges might be skipped.")

# -------------------------------
# 🗓️ SAFE DATE PARSING & SORTING (YYYY-MM-DD chronological)
# -------------------------------
if date_col:
    schedule_df["__Date_parsed"] = pd.to_datetime(
        schedule_df[date_col].astype(str).str.strip(),
        errors="coerce",
        infer_datetime_format=True,
        dayfirst=False
    )
else:
    # fallback: try first column
    schedule_df["__Date_parsed"] = pd.to_datetime(
        schedule_df.iloc[:, 0].astype(str).str.strip(),
        errors="coerce",
        infer_datetime_format=True,
        dayfirst=False
    )

# If too many NA dates, try dayfirst fallback on the named column
if schedule_df["__Date_parsed"].isna().mean() > 0.25 and date_col:
    schedule_df["__Date_parsed"] = pd.to_datetime(
        schedule_df[date_col].astype(str).str.strip(),
        errors="coerce",
        infer_datetime_format=True,
        dayfirst=True
    )

schedule_df = schedule_df.dropna(subset=["__Date_parsed"])
schedule_df = schedule_df.sort_values("__Date_parsed").reset_index(drop=True)

# -------------------------
# PREP All-Stats features (use all numeric columns)
# -------------------------
numeric_team_cols = all_stats.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_team_cols) == 0:
    st.error("No numeric columns detected in All_Stats-THE_TABLE.csv; cannot build numeric features.")
    st.stop()

cat_team_cols = []
if all_stats_coach_col:
    cat_team_cols.append(all_stats_coach_col)
if all_stats_conf_col:
    cat_team_cols.append(all_stats_conf_col)

# Build lookup dicts for numeric and categorical team info
team_numeric_lookup = {}
team_cat_lookup = {}
for _, r in all_stats.iterrows():
    tname = r[all_stats_team_col]
    if pd.isna(tname):
        continue
    key = str(tname).strip()
    # numeric vector
    try:
        numeric_vec = r[numeric_team_cols].astype(float).fillna(0.0).values
    except Exception:
        # fallback: convert elementwise to numeric safely
        numeric_vec = pd.to_numeric(r[numeric_team_cols], errors="coerce").fillna(0.0).values
    team_numeric_lookup[key] = numeric_vec
    # categorical map
    team_cat_lookup[key] = {}
    for c in cat_team_cols:
        team_cat_lookup[key][c] = r[c] if c in all_stats.columns else np.nan

st.write(f"Using {len(numeric_team_cols)} numeric team-level stats from All_Stats.")

# -------------------------
# BUILD TRAINING SET FROM daily_df
# -------------------------
daily_numeric_cols = daily_df.select_dtypes(include=[np.number]).columns.tolist()

# Helper to interpret HAN-like values into Home/Away/Neutral
def interpret_han_val(v):
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

train_rows = []
skip_count = 0
for idx, r in daily_df.iterrows():
    # get team/opp names (if available)
    if daily_team_col in daily_df.columns:
        tname = str(r[daily_team_col]).strip()
    else:
        tname = None
    if daily_opp_col in daily_df.columns:
        oname = str(r[daily_opp_col]).strip()
    else:
        oname = None

    if not tname or not oname:
        skip_count += 1
        continue

    # both must exist in all_stats lookup
    if tname not in team_numeric_lookup or oname not in team_numeric_lookup:
        skip_count += 1
        continue

    # numeric features: team, opp, diff
    tnum = team_numeric_lookup[tname]
    onum = team_numeric_lookup[oname]
    feat = np.concatenate([tnum, onum, tnum - onum])

    # include daily numeric features
    if len(daily_numeric_cols) > 0:
        try:
            daily_feats = r[daily_numeric_cols].astype(float).fillna(0.0).values
            feat = np.concatenate([feat, daily_feats])
        except Exception:
            # if conversion fails, append zeros for those columns
            feat = np.concatenate([feat, np.zeros(len(daily_numeric_cols))])

    # home flag (if daily has HAN column)
    home_flag = 0
    if daily_han_col and daily_han_col in daily_df.columns:
        han_val = interpret_han_val(r[daily_han_col])
        if han_val == "Home":
            home_flag = 1
    # non-conf flag (if daily has a column or infer from conferences)
    nonconf_flag = 0
    if daily_nonconf_col and daily_nonconf_col in daily_df.columns:
        v = r[daily_nonconf_col]
        if not pd.isna(v) and (str(v).strip() in ("1", "True", "TRUE", "YES", "Yes", "Y")):
            nonconf_flag = 1
    else:
        # infer by comparing conferences from all_stats if available
        tconf = team_cat_lookup.get(tname, {}).get(all_stats_conf_col) if all_stats_conf_col else None
        oconf = team_cat_lookup.get(oname, {}).get(all_stats_conf_col) if all_stats_conf_col else None
        if tconf is not None and oconf is not None and str(tconf).strip() != str(oconf).strip():
            nonconf_flag = 1

    feat = np.concatenate([feat, np.array([home_flag, nonconf_flag])])

    # categorical coach/conf values (team coach, opp coach, team conf, opp conf)
    cat_vals = []
    for c in cat_team_cols:
        cat_vals.append(str(team_cat_lookup.get(tname, {}).get(c, "")))
        cat_vals.append(str(team_cat_lookup.get(oname, {}).get(c, "")))

    # target margin and win
    if "Points" in daily_df.columns and "Opp Points" in daily_df.columns:
        try:
            margin = float(r["Points"]) - float(r["Opp Points"])
            win = 1 if margin > 0 else 0
        except Exception:
            margin = np.nan
            win = np.nan
    else:
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

# Build numeric matrix and targets
if len(train_rows) == 0:
    st.error("No valid training rows constructed — check team name consistency between daily predictor and all_stats.")
    st.stop()

X_num = np.vstack([r["feat"] for r in train_rows])
y_margin = np.array([r["margin"] for r in train_rows])
y_win = np.array([r["win"] for r in train_rows])

# Categorical encoding for coach/conf
if len(cat_team_cols) > 0:
    cat_list = [r["cats"] for r in train_rows]
    cat_matrix = np.array(cat_list)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    try:
        cat_encoded = enc.fit_transform(cat_matrix)
    except Exception:
        cat_encoded = np.zeros((len(train_rows), cat_matrix.shape[1]))
else:
    cat_encoded = np.zeros((len(train_rows), 0))

# Combine numeric + cat
if cat_encoded.shape[1] > 0:
    X_full = np.hstack([X_num, cat_encoded])
else:
    X_full = X_num

# Replace NaN/inf with zeros
X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

# Filter out rows where targets are NaN
valid_mask = ~np.isnan(y_margin) & ~np.isnan(y_win)
X_full = X_full[valid_mask]
y_margin = y_margin[valid_mask]
y_win = y_win[valid_mask]

if X_full.shape[0] < 10:
    st.warning(f"After cleaning there are only {X_full.shape[0]} usable training rows. Consider adding more historical rows or ensuring team matching.")

# -------------------------
# MODEL TRAINING
# -------------------------
st.write(f"Training models on {X_full.shape[0]} examples with {X_full.shape[1]} features...")

clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_full, y_win)

reg = RandomForestRegressor(n_estimators=300, random_state=42)
reg.fit(X_full, y_margin)

# residual std for simple CIs
pred_train_margin = reg.predict(X_full)
resid_sd = float(np.std(y_margin - pred_train_margin)) if X_full.shape[0] > 0 else 10.0
st.write(f"Trained models. Margin residual sd ≈ {resid_sd:.2f}")

# -------------------------
# SCHEDULE & TEAM DROPDOWN
# -------------------------
team_values = sorted(all_stats[all_stats_team_col].dropna().unique().astype(str).tolist())
default_index = team_values.index(INITIAL_TEAM) if INITIAL_TEAM in team_values else 0
selected_team = st.selectbox("Select a team (default controlled by top-of-file INITIAL_TEAM)", team_values, index=default_index)
st.write(f"Default INITIAL_TEAM variable is: '{INITIAL_TEAM}' (edit file to change)")

# Filter schedule so we only use rows where the team column equals the selected team
mask = schedule_df[schedule_team_col].astype(str).str.strip() == selected_team
selected_schedule = schedule_df.loc[mask].copy().sort_values("__Date_parsed").reset_index(drop=True)

if selected_schedule.empty:
    st.info(f"No scheduled games found for {selected_team}.")
    st.stop()

# How many upcoming games to show
N = st.number_input("How many upcoming games to show", min_value=1, max_value=12, value=3, step=1)

# -------------------------
# PREDICT FOR EACH UPCOMING GAME (honoring HAN and non-conf)
# -------------------------
pred_rows = []

for _, game in selected_schedule.head(N).iterrows():
    row_team = str(game[schedule_team_col]).strip()
    row_opp = str(game[schedule_opp_col]).strip()

    # HAN / location detection (prefer explicit column)
    location = None
    han_val = None
    if han_col and han_col in game.index:
        han_val = interpret_han_val(game[han_col])
        location = han_val

    # If HAN not present or unclear, determine by which column equals selected_team
    if location is None:
        if row_team == selected_team:
            location = "Home"
        elif row_opp == selected_team:
            location = "Away"
        else:
            location = "Neutral"

    # Determine which name is opponent relative to selection
    if row_team == selected_team:
        opp_name = row_opp
    elif row_opp == selected_team:
        opp_name = row_team
    else:
        # fallback
        opp_name = row_opp

    # ensure both exist in lookup
    if selected_team not in team_numeric_lookup or opp_name not in team_numeric_lookup:
        st.write(f"Warning: stats for {selected_team} or {opp_name} missing; skipping.")
        continue

    # build numeric features: team, opp, diff
    team_num = team_numeric_lookup[selected_team]
    opp_num = team_numeric_lookup[opp_name]
    feat = np.concatenate([team_num, opp_num, team_num - opp_num])

    # home flag
    home_flag = 1 if location == "Home" else 0
    # nonconf flag: check schedule's nonconf column or infer from conferences
    nonconf_flag = 0
    if nonconf_col and nonconf_col in schedule_df.columns:
        v = game.get(nonconf_col)
        try:
            if not pd.isna(v) and str(v).strip().upper() in ("1", "TRUE", "YES", "Y"):
                nonconf_flag = 1
        except Exception:
            nonconf_flag = 0
    else:
        # infer by conferences if available
        if all_stats_conf_col:
            tconf = team_cat_lookup.get(selected_team, {}).get(all_stats_conf_col)
            oconf = team_cat_lookup.get(opp_name, {}).get(all_stats_conf_col)
            if tconf is not None and oconf is not None and str(tconf).strip() != str(oconf).strip():
                nonconf_flag = 1

    feat = np.concatenate([feat, np.array([home_flag, nonconf_flag])])

    # categorical vector for this matchup
    cat_vals = []
    for c in cat_team_cols:
        cat_vals.append(str(team_cat_lookup.get(selected_team, {}).get(c, "")))
        cat_vals.append(str(team_cat_lookup.get(opp_name, {}).get(c, "")))

    if len(cat_vals) > 0:
        try:
            cat_enc = enc.transform([cat_vals])
        except Exception:
            # unknown/novel categories -> encode as -1
            cat_enc = np.array([[-1] * len(cat_vals)])
        X_future = np.hstack([feat, cat_enc.ravel()])
    else:
        X_future = feat

    X_future = np.nan_to_num(X_future, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)

    # Align dims with training (truncate / pad)
    if X_future.shape[1] != X_full.shape[1]:
        min_cols = min(X_future.shape[1], X_full.shape[1])
        X_tmp = np.zeros((1, X_full.shape[1]))
        X_tmp[0, :min_cols] = X_future[0, :min_cols]
        X_future = X_tmp

    # predictions
    win_prob = clf.predict_proba(X_future)[0][1] if hasattr(clf, "predict_proba") else float(clf.predict(X_future)[0])
    pred_margin = float(reg.predict(X_future)[0])

    # baseline avg total from training (use daily Points + Opp Points if available)
    avg_total = 140.0
    if "Points" in daily_df.columns and "Opp Points" in daily_df.columns:
        avg_total = float((pd.to_numeric(daily_df["Points"], errors="coerce").fillna(0) + pd.to_numeric(daily_df["Opp Points"], errors="coerce").fillna(0)).mean())

    # Convert margin to team/opp scores (team is selected_team)
    pred_team_score = (avg_total + pred_margin) / 2.0
    pred_opp_score = (avg_total - pred_margin) / 2.0

    # 95% CI for margin -> convert to score CI
    z = 1.96
    margin_lo = pred_margin - z * resid_sd
    margin_hi = pred_margin + z * resid_sd

    team_lo = (avg_total + margin_lo) / 2.0
    team_hi = (avg_total + margin_hi) / 2.0
    opp_lo = (avg_total - margin_hi) / 2.0
    opp_hi = (avg_total - margin_lo) / 2.0

    pred_rows.append({
        "Date": game["__Date_parsed"].strftime("%Y-%m-%d") if not pd.isna(game["__Date_parsed"]) else str(game.get(date_col, "")),
        "Opponent": opp_name,
        "HAN (Location)": location,
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
