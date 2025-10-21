# APP/Pages/5_Schedule_Predictor.py
"""
Schedule Predictor (Schedule → Predictions)

What changed (ML):
- Still encodes categorical coach/conference and uses ALL numeric team stats from All_Stats.
- Re-trains on each run using Daily predictor *results*:
    * RandomForestClassifier -> Win probability (y_win from Points - Opp Points)
    * MultiOutputRegressor(RandomForestRegressor) -> [Points, Opp Points] directly
- Predictions for upcoming schedule rows are produced as:
    Projected: {SelectedTeam PredPoints} - {Opponent PredOppPoints}
    Win% from classifier; Margin = PredPoints - PredOppPoints

Other:
- Dates parsed with explicit MM/DD/YYYY first (your files use US-style), with robust fallback.
- Only rows where schedule Team == selected team are used (no mirrored duplicate counts).
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# -------------------------
# CONFIG: change default here
# -------------------------
INITIAL_TEAM = "Wisconsin"   # <- edit this to change default selected team

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
    "Select a team to view upcoming games. Models train on Daily results and predict your team's points & opponent points using All_Stats numeric features plus categorical coach/conference and game flags."
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
# HELPERS
# -------------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def parse_mmddyyyy(series_like):
    """
    Parse US-style dates first (MM/DD/YYYY). If that fails broadly, fallback to standard parser.
    """
    s = series_like.astype(str).str.strip()
    # First: strict MM/DD/YYYY if it matches pattern
    parsed_try = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    if parsed_try.isna().mean() < 0.75:
        return parsed_try
    # Fallback: generic parser (no dayfirst)
    parsed_generic = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if parsed_generic.isna().mean() < 0.75:
        return parsed_generic
    # Last resort: allow dayfirst (if user drops in a diff csv)
    parsed_dayfirst = pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
    return parsed_dayfirst

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

POWER_CONF_SET = {
    "SEC", "ACC", "BIG TEN", "B1G", "BIG 12", "BIG12", "PAC-12", "PAC 12", "BIG EAST"
}

def is_power_conf(x):
    if pd.isna(x):
        return 0.0
    s = str(x).strip().upper()
    return 1.0 if s in POWER_CONF_SET else 0.0

# -------------------------
# DETECT KEY COLUMNS
# -------------------------
# schedule
schedule_team_col = find_col(schedule_df, ["Team", "Teams", "team", "Home", "Home Team"])
schedule_opp_col  = find_col(schedule_df, ["Opponent", "Opp", "opponent", "Away", "Away Team"])
schedule_date_col = find_col(schedule_df, ["Date", "date", "Game_Date", "Game Date"])
schedule_han_col  = find_col(schedule_df, ["HAN", "Han", "Home/Away", "Location", "Loc", "HomeAway", "HOME/ AWAY", "HOME/AWAY"])
schedule_conf_col = find_col(schedule_df, ["Conference", "Conf", "conference"])
schedule_oppconf_col = find_col(schedule_df, ["Opponent Conference", "Opp Conference", "OpponentConference"])
schedule_nonconf_col = find_col(schedule_df, ["Non Conference Game", "NonConference", "Non Conf", "NonConf", "Non-Conference"])

if schedule_team_col is None or schedule_opp_col is None:
    st.error("Could not detect 'Team' and 'Opponent' columns in schedule file.")
    st.stop()

# all_stats
all_stats_team_col  = find_col(all_stats, ["Teams", "Team", "team"])
all_stats_coach_col = find_col(all_stats, ["Coach Name", "Coach", "Coach_Name", "coach"])
all_stats_conf_col  = find_col(all_stats, ["Conference", "Conf", "conference"])

if all_stats_team_col is None:
    st.error("Could not detect team column in All_Stats-THE_TABLE.csv (expected 'Teams' or 'Team').")
    st.stop()

# daily
daily_team_col = find_col(daily_df, ["Team", "Teams", "team"])
daily_opp_col  = find_col(daily_df, ["Opponent", "Opp", "opponent"])
daily_points_col = find_col(daily_df, ["Points", "Points "])
daily_opp_points_col = find_col(daily_df, ["Opp Points", "Opp Points ", "OppPoints", "Opp_Points"])
daily_han_col  = find_col(daily_df, ["HAN", "Han", "Home/Away", "Location", "Loc"])
daily_conf_col = find_col(daily_df, ["Conference", "Conf", "conference"])
daily_oppconf_col = find_col(daily_df, ["Opponent Conference", "Opp Conference", "OpponentConference", "Opp_Conference"])
daily_nonconf_col = find_col(daily_df, ["Non Conference Game", "NonConference", "Non Conf", "NonConf", "Non-Conference"])
daily_coach_col = find_col(daily_df, ["Coach Name", "Coach", "Coach_Name", "coach"])
daily_oppcoach_col = find_col(daily_df, ["Opponent Coach", "Opp Coach", "Opp_Coach"])

if daily_team_col is None or daily_opp_col is None or daily_points_col is None or daily_opp_points_col is None:
    st.error("Daily predictor must include Team, Opponent, Points, and Opp Points columns.")
    st.stop()

# -------------------------
# DATE PARSING & SORTING
# -------------------------
# schedule parsing with MM/DD/YYYY priority
if schedule_date_col is None:
    schedule_date_col = schedule_df.columns[0]
schedule_df["__Date_parsed"] = parse_mmddyyyy(schedule_df[schedule_date_col])
schedule_df = schedule_df.dropna(subset=["__Date_parsed"]).copy()
schedule_df = schedule_df.sort_values("__Date_parsed").reset_index(drop=True)

# -------------------------
# PREP ALL_STATS LOOKUPS
# -------------------------
numeric_team_cols = all_stats.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_team_cols) == 0:
    st.error("No numeric columns detected in All_Stats-THE_TABLE.csv; cannot build numeric features.")
    st.stop()

team_numeric_lookup = {}
team_cat_lookup = {}

for _, r in all_stats.iterrows():
    tname = r[all_stats_team_col]
    if pd.isna(tname):
        continue
    key = str(tname).strip()
    # numeric vector (fill NaNs with 0)
    numeric_vec = pd.to_numeric(r[numeric_team_cols], errors="coerce").fillna(0.0).values
    team_numeric_lookup[key] = numeric_vec
    # minimal categorical map (coach/conf for the team)
    team_cat_lookup[key] = {
        "coach": str(r.get(all_stats_coach_col, "NA")) if all_stats_coach_col else "NA",
        "conf":  str(r.get(all_stats_conf_col, "NA"))  if all_stats_conf_col  else "NA",
    }

st.write(f"Using {len(numeric_team_cols)} numeric team-level features from All_Stats.")

# -------------------------
# BUILD TRAINING SET FROM DAILY (features from All_Stats + flags + categorical)
# -------------------------
train_rows = []
skip = 0

for _, r in daily_df.iterrows():
    tname = str(r[daily_team_col]).strip() if pd.notna(r[daily_team_col]) else None
    oname = str(r[daily_opp_col]).strip() if pd.notna(r[daily_opp_col]) else None
    if not tname or not oname:
        skip += 1
        continue
    if tname not in team_numeric_lookup or oname not in team_numeric_lookup:
        skip += 1
        continue

    # numeric: team, opp, and diff
    tnum = team_numeric_lookup[tname]
    onum = team_numeric_lookup[oname]
    base_vec = np.concatenate([tnum, onum, tnum - onum], dtype=float)

    # flags
    # home (from daily HAN if present)
    home_flag = 0.0
    if daily_han_col and (daily_han_col in r.index):
        loc = interpret_han_val(r[daily_han_col])
        home_flag = 1.0 if loc == "Home" else 0.0

    # non-conf
    non_conf_flag = 0.0
    if daily_nonconf_col and (daily_nonconf_col in r.index):
        v = r[daily_nonconf_col]
        try:
            non_conf_flag = 1.0 if (str(v).strip().upper() in ("1", "TRUE", "YES", "Y")) else 0.0
        except Exception:
            non_conf_flag = 0.0
    else:
        # infer by comparing confs if present on daily, otherwise via All_Stats lookup
        tconf = r.get(daily_conf_col) if daily_conf_col else team_cat_lookup[tname]["conf"]
        oconf = r.get(daily_oppconf_col) if daily_oppconf_col else team_cat_lookup[oname]["conf"]
        try:
            if pd.notna(tconf) and pd.notna(oconf) and str(tconf).strip().upper() != str(oconf).strip().upper():
                non_conf_flag = 1.0
        except Exception:
            non_conf_flag = 0.0

    # power-conf flags
    tconf_use = r.get(daily_conf_col) if daily_conf_col else team_cat_lookup[tname]["conf"]
    oconf_use = r.get(daily_oppconf_col) if daily_oppconf_col else team_cat_lookup[oname]["conf"]
    team_power = is_power_conf(tconf_use)
    opp_power  = is_power_conf(oconf_use)

    flags_vec = np.array([home_flag, non_conf_flag, team_power, opp_power], dtype=float)

    # categorical tokens for encoder (team, opp, coachs, confs)
    tcoach = str(r.get(daily_coach_col, team_cat_lookup[tname]["coach"])) if daily_coach_col else team_cat_lookup[tname]["coach"]
    ocoach = str(r.get(daily_oppcoach_col, team_cat_lookup[oname]["coach"])) if daily_oppcoach_col else team_cat_lookup[oname]["coach"]
    tconf  = str(tconf_use) if pd.notna(tconf_use) else "NA"
    oconf  = str(oconf_use) if pd.notna(oconf_use) else "NA"

    cats = [tname, oname, tcoach, ocoach, tconf, oconf]  # keep this exact order

    # targets
    pts = pd.to_numeric(r[daily_points_col], errors="coerce")
    opp = pd.to_numeric(r[daily_opp_points_col], errors="coerce")
    if pd.isna(pts) or pd.isna(opp):
        skip += 1
        continue
    win = 1 if (pts - opp) > 0 else 0

    train_rows.append({
        "base": base_vec,
        "flags": flags_vec,
        "cats": cats,
        "y_points": float(pts),
        "y_opp": float(opp),
        "y_win": win
    })

if len(train_rows) < 20:
    st.warning(f"Only {len(train_rows)} usable training rows built from daily → all_stats. Training may be weak until more results accumulate.")

if len(train_rows) == 0:
    st.error("No valid training rows. Ensure daily team names match All_Stats team names.")
    st.stop()

# numeric stacks
X_base = np.vstack([r["base"] for r in train_rows])                  # [team, opp, diff]
X_flags = np.vstack([r["flags"] for r in train_rows])                # [home, nonconf, team_power, opp_power]

# categorical encoding
cat_matrix = np.array([r["cats"] for r in train_rows], dtype=object)
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_cats = enc.fit_transform(cat_matrix)

# final X for training
X_train = np.hstack([X_base, X_flags, X_cats])
X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

Y_points_opp = np.column_stack([ [r["y_points"] for r in train_rows],
                                 [r["y_opp"]    for r in train_rows] ]).astype(float)
y_win = np.array([r["y_win"] for r in train_rows], dtype=int)

# -------------------------
# TRAIN MODELS
# -------------------------
st.write(f"Training models on {X_train.shape[0]} examples with {X_train.shape[1]} features...")

# Classifier for win probability
clf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt",
    min_samples_leaf=1
)
clf.fit(X_train, y_win)

# Multi-output regressor for [Points, Opp Points]
rf_multi = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        min_samples_leaf=1
    )
)
rf_multi.fit(X_train, Y_points_opp)

st.success("Models trained.")

# -------------------------
# TEAM SELECTION & SCHEDULE FILTER
# -------------------------
team_values = sorted(all_stats[all_stats_team_col].dropna().astype(str).unique().tolist())
default_index = team_values.index(INITIAL_TEAM) if INITIAL_TEAM in team_values else 0
selected_team = st.selectbox("Select a team (default controlled by top-of-file INITIAL_TEAM)", team_values, index=default_index)
st.write(f"Default INITIAL_TEAM variable is: '{INITIAL_TEAM}' (edit file to change)")

# Only rows where schedule Team == selected team (prevents mirrored duplicates)
mask = schedule_df[schedule_team_col].astype(str).str.strip() == selected_team
selected_schedule = schedule_df.loc[mask].copy().sort_values("__Date_parsed").reset_index(drop=True)

if selected_schedule.empty:
    st.info(f"No scheduled games found for {selected_team}.")
    st.stop()

N = st.number_input("How many upcoming games to show", min_value=1, max_value=20, value=3, step=1)

# -------------------------
# FEATURE BUILDER FOR FUTURE GAMES (mirrors training recipe)
# -------------------------
def build_feature_vector_for_match(team_name: str, opp_name: str, location_val, team_conf_val, opp_conf_val, opp_coach_val=None, team_coach_val=None):
    """
    Build a single X vector using the same structure as training:
    [team_stats, opp_stats, diff, flags(home, nonconf, team_power, opp_power), cats(team, opp, tcoach, ocoach, tconf, oconf)]
    """
    # numeric
    if (team_name not in team_numeric_lookup) or (opp_name not in team_numeric_lookup):
        return None  # caller will skip

    tnum = team_numeric_lookup[team_name]
    onum = team_numeric_lookup[opp_name]
    base_vec = np.concatenate([tnum, onum, tnum - onum], dtype=float)

    # flags
    home_flag = 1.0 if interpret_han_val(location_val) == "Home" else 0.0

    # non-conf (from schedule conf fields if provided, else infer via All_Stats)
    tconf = team_conf_val if team_conf_val is not None else team_cat_lookup[team_name]["conf"]
    oconf = opp_conf_val  if opp_conf_val  is not None else team_cat_lookup[opp_name]["conf"]

    non_conf_flag = 0.0
    try:
        if pd.notna(tconf) and pd.notna(oconf) and str(tconf).strip().upper() != str(oconf).strip().upper():
            non_conf_flag = 1.0
    except Exception:
        non_conf_flag = 0.0

    team_power = is_power_conf(tconf)
    opp_power  = is_power_conf(oconf)
    flags_vec = np.array([home_flag, non_conf_flag, team_power, opp_power], dtype=float)

    # cats in the same order we trained on
    tcoach = (str(team_coach_val) if team_coach_val is not None else team_cat_lookup[team_name]["coach"]) or "NA"
    ocoach = (str(opp_coach_val)  if opp_coach_val  is not None else team_cat_lookup[opp_name]["coach"])   or "NA"
    tconf_s = str(tconf) if pd.notna(tconf) else "NA"
    oconf_s = str(oconf) if pd.notna(oconf) else "NA"

    cats = [team_name, opp_name, tcoach, ocoach, tconf_s, oconf_s]
    cats_enc = enc.transform([cats])  # shape (1, 6)

    # final vect
    X_vec = np.hstack([base_vec, flags_vec, cats_enc.ravel()])
    X_vec = np.nan_to_num(X_vec, nan=0.0, posinf=0.0, neginf=0.0)
    return X_vec.reshape(1, -1)

# -------------------------
# PREDICT PER UPCOMING GAME
# -------------------------
pred_rows = []

for _, game in selected_schedule.head(N).iterrows():
    # Opponent & metadata
    opp_name = str(game[schedule_opp_col]).strip()

    # Location
    loc_val = game.get(schedule_han_col) if schedule_han_col and schedule_han_col in game.index else None

    # Conf & Coaches (best-effort from schedule; fallback to All_Stats mappings inside builder)
    tconf_val = game.get(schedule_conf_col) if schedule_conf_col and schedule_conf_col in game.index else None
    oconf_val = game.get(schedule_oppconf_col) if schedule_oppconf_col and schedule_oppconf_col in game.index else None

    team_coach_val = game.get("Coach Name") if "Coach Name" in game.index else game.get("Coach")
    opp_coach_val  = game.get("Opponent Coach") if "Opponent Coach" in game.index else game.get("Opp Coach")

    # Build feature vector using same structure as training
    X_future = build_feature_vector_for_match(
        team_name=selected_team,
        opp_name=opp_name,
        location_val=loc_val,
        team_conf_val=tconf_val,
        opp_conf_val=oconf_val,
        opp_coach_val=opp_coach_val,
        team_coach_val=team_coach_val
    )
    if X_future is None:
        st.write(f"Warning: Missing All_Stats numeric for {selected_team} or {opp_name}; skipping.")
        continue

    # Align shape if needed (defensive padding/trunc)
    if X_future.shape[1] != X_train.shape[1]:
        cols = min(X_future.shape[1], X_train.shape[1])
        tmp = np.zeros((1, X_train.shape[1]), dtype=float)
        tmp[:, :cols] = X_future[:, :cols]
        X_future = tmp

    # Predict [Points, Opp Points] and win probability
    pred_points, pred_opp_points = rf_multi.predict(X_future)[0]
    pred_points = max(0.0, float(pred_points))
    pred_opp_points = max(0.0, float(pred_opp_points))
    pred_margin = pred_points - pred_opp_points

    win_prob = float(clf.predict_proba(X_future)[0][1]) if hasattr(clf, "predict_proba") else float(clf.predict(X_future)[0])

    # Location clean label
    loc_label = interpret_han_val(loc_val)
    if loc_label is None:
        # fallback: if selected team is in the Team col, it's a "Home" row in your file schema; else unknown
        loc_label = "Home"

    # Date string
    d = game["__Date_parsed"]
    date_str = d.strftime("%Y-%m-%d") if pd.notna(d) else str(game.get(schedule_date_col, ""))

    pred_rows.append({
        "Date": date_str,
        "Opponent": opp_name,
        "HAN (Location)": loc_label,
        "Win_Prob": f"{win_prob*100:.1f}%",
        "Predicted_Team_Points": int(round(pred_points)),
        "Predicted_Opp_Points": int(round(pred_opp_points)),
        "Predicted_Margin": round(pred_margin, 1),
        "Projected": f"{selected_team} {int(round(pred_points))} - {opp_name} {int(round(pred_opp_points))}",
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
    "- Models retrain each run from Daily predictor: classifier for Win% and a multi-output regressor for [Points, Opp Points].\n"
    "- Features at training and prediction time use the same structure: All_Stats numeric vectors (team, opponent, diff) + flags (home, non-conf, power-conf) + categorical encodings (team, opponent, coach, conference).\n"
    "- Dates are parsed as MM/DD/YYYY first (per your files), then fallback parsers if needed.\n"
    "- This keeps your original page layout and improves prediction granularity (scores vary by matchup)."
)
