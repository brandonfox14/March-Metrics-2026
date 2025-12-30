# APP/Pages/9_Betting.py
# =========================================================
# Betting Page — multi-market plan (ML/Spread/OU)
# =========================================================
#493 is where to adjust things
import os
import math
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

# =========================================================
# CONFIG
# =========================================================
BASE = "Data/26_March_Madness_Databook"
SCHEDULE_FILE = os.path.join(BASE, "2026 Schedule Transfer-Table 1.csv")
DAILY_FILE    = os.path.join(BASE, "Daily_predictor_data-Table 1.csv")

st.set_page_config(page_title="Betting", layout="wide")
st.title("Betting")

# =========================================================
# HELPERS
# =========================================================
def load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        return None
    try:
        df = pd.read_csv(path, encoding="latin1")
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error reading {path}: {e}")
        return None

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def parse_mdy(series_like: pd.Series) -> pd.Series:
    s = series_like.astype(str).str.strip()
    out = pd.to_datetime(s, errors="coerce", format="%m/%d/%Y")
    mask = out.isna()
    if mask.any():
        try2 = pd.to_datetime(s[mask], errors="coerce", infer_datetime_format=True)
        out.loc[mask] = try2
    return out

def american_to_decimal(odds) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 1.0 + (o / 100.0)
    elif o < 0:
        return 1.0 + (100.0 / -o)
    return np.nan

def payout_if_win_total(stake: float, odds_us: float) -> float:
    """Total return if win = stake + winnings."""
    if not np.isfinite(stake) or stake <= 0 or not np.isfinite(odds_us):
        return np.nan
    dec = american_to_decimal(odds_us)
    if not np.isfinite(dec):
        return np.nan
    return float(stake * dec)

def safe_num(x, default=np.nan) -> float:
    try:
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default

def uniq_jitter(key: str, scale: float = 0.11) -> float:
    h = abs(hash(key)) % 10_000
    return (h / 10_000.0 - 0.5) * 2.0 * scale  # [-scale, +scale]

def clean_rank_value(x) -> float:
    try:
        v = float(x)
        if np.isnan(v): return 51.0
        return v
    except Exception:
        return 51.0

def extract_top50_any(row: Optional[pd.Series]) -> List[Tuple[str, int]]:
    if row is None:
        return []
    out = []
    for c in row.index:
        cu = c.strip()
        cu_upper = cu.upper()
        if ("RANK" in cu_upper) or cu_upper.endswith("_RANK"):
            rv = clean_rank_value(row.get(c))
            if rv <= 50:
                out.append((c, int(rv)))
    return sorted(out, key=lambda z: z[1])

def is_penn_state(name: str) -> bool:
    s = str(name).strip().upper()
    return ("PENN ST" in s) or ("PENN STATE" in s)

def clamp_prob(p: float, lo: float = 0.005, hi: float = 0.995) -> float:
    if not np.isfinite(p):
        return np.nan
    return float(min(hi, max(lo, p)))

def pick_binary_side(p_event: float, label_event: str, label_other: str):
    """
    Given P(event), return (picked_label, picked_prob).
    Ensures picked_prob >= 0.50 (binary complement).
    """
    p_event = clamp_prob(p_event)
    if not np.isfinite(p_event):
        return None, np.nan
    p_other = 1.0 - p_event
    if p_event >= 0.5:
        return label_event, p_event
    return label_other, p_other

def enforce_ml_spread_consistency(line: float, win_prob: float, cover_prob: float) -> float:
    """
    Enforce logical relationship between Win and Cover probabilities for a given side.
    - If line < 0 (favorite laying points): P(cover) <= P(win)
    - If line > 0 (underdog getting points): P(cover) >= P(win)
    - If line == 0: P(cover) == P(win)
    """
    if not (np.isfinite(line) and np.isfinite(win_prob) and np.isfinite(cover_prob)):
        return cover_prob
    win_prob = clamp_prob(win_prob)
    cover_prob = clamp_prob(cover_prob)

    if line < 0:
        return float(min(cover_prob, win_prob))
    if line > 0:
        return float(max(cover_prob, win_prob))
    return float(win_prob)

def truthy_flag(v) -> bool:
    if pd.isna(v):
        return False
    s = str(v).strip().upper()
    return s not in ("0", "", "N", "NO", "FALSE", "F", "NA", "NAN", "NONE")

# Goal-based ranking helpers
def profit_multiple(dec: float) -> float:
    # profit per $1 stake if win
    if not np.isfinite(dec) or dec <= 1.0:
        return np.nan
    return float(dec - 1.0)

def expected_profit_per_dollar(prob: float, dec: float) -> float:
    # EV profit per $1 stake (not total return)
    # = p*(dec-1) - (1-p)
    if not np.isfinite(prob) or not np.isfinite(dec) or dec <= 1.0:
        return np.nan
    p = clamp_prob(prob)
    return float(p * (dec - 1.0) - (1.0 - p))

# =========================================================
# LOAD DATA
# =========================================================
schedule_df_raw = load_csv(SCHEDULE_FILE)
schedule_df = None if schedule_df_raw is None else schedule_df_raw.copy()
daily_df    = load_csv(DAILY_FILE)
if schedule_df is None or daily_df is None:
    st.stop()

# =========================================================
# COLUMN DETECTION (SCHEDULE)
# =========================================================
team_col   = find_col(schedule_df, ["Team", "Teams"])
opp_col    = find_col(schedule_df, ["Opponent", "Opp", "opponent"])
date_col   = find_col(schedule_df, ["Date", "Game Date", "Game_Date"])
han_col    = find_col(schedule_df, ["HAN","Home/Away","HomeAway","Location","Loc"])
conf_col   = find_col(schedule_df, ["Conference"])
opp_conf_col = find_col(schedule_df, ["Opponent Conference","Opp Conference"])
coach_col  = find_col(schedule_df, ["Coach Name","Coach","Coach_Name"])
opp_coach_col = find_col(schedule_df, ["Opponent Coach","Opp Coach"])

line_col   = find_col(schedule_df, ["Line","Spread","Vegas Line"])
ml_col     = find_col(schedule_df, ["ML","Moneyline","Money Line"])
ou_col     = find_col(schedule_df, ["Over/Under Line","OverUnder","Over Under Line","O/U","Total Points Line","Total"])

top25_col  = find_col(schedule_df, ["Top 25 Opponent","Top25","Top 25"])
mm_col     = find_col(schedule_df, ["March Madness Opponent","March Madness"])

if team_col is None or opp_col is None or date_col is None:
    st.error("Schedule must include Team, Opponent, and Date.")
    st.stop()

schedule_df["__Date"] = parse_mdy(schedule_df[date_col])
schedule_df = schedule_df.dropna(subset=["__Date"]).copy()

# Deduplicate mirrored fixtures for scoring
seen = set()
keep_idx = []
for idx, r in schedule_df.iterrows():
    t = str(r[team_col]).strip()
    o = str(r[opp_col]).strip()
    d = r["__Date"].date()
    key = tuple(sorted([t.lower(), o.lower()])) + (d,)
    if key in seen:
        continue
    seen.add(key)
    keep_idx.append(idx)
schedule_df = schedule_df.loc[keep_idx].reset_index(drop=True)

# =========================================================
# DAILY FEATURE SPACE & TARGETS
# =========================================================
d_team = find_col(daily_df, ["Team","Teams","team"])
d_opp  = find_col(daily_df, ["Opponent","Opp","opponent"])

d_pts  = find_col(daily_df, ["Points"])
d_opp_pts = find_col(daily_df, ["Opp Points","Opp_Points","OppPoints"])
d_line = find_col(daily_df, ["Line"])
d_ou   = find_col(daily_df, ["Over/Under Line","OverUnder","Over Under Line","O/U"])
d_ml   = find_col(daily_df, ["ML","Moneyline","Money Line"])

if d_team is None or d_opp is None or d_pts is None or d_opp_pts is None:
    st.error("Daily predictor must include Team, Opponent, Points, Opp Points.")
    st.stop()

# numeric columns (+ objects coercible to numeric)
daily_numeric_cols = daily_df.select_dtypes(include=[np.number]).columns.tolist()
obj_as_num = []
for c in daily_df.columns:
    if c not in daily_numeric_cols:
        try:
            pd.to_numeric(daily_df[c], errors="raise")
            obj_as_num.append(c)
        except Exception:
            pass
daily_numeric_cols = list(dict.fromkeys(daily_numeric_cols + obj_as_num))

# Drop pure point targets from feature set
for drop_c in [d_pts, d_opp_pts]:
    if drop_c in daily_numeric_cols:
        daily_numeric_cols.remove(drop_c)

# Coerce points
daily_df[d_pts]     = pd.to_numeric(daily_df[d_pts], errors="coerce")
daily_df[d_opp_pts] = pd.to_numeric(daily_df[d_opp_pts], errors="coerce")

# Targets: SM and Total
sm_col = find_col(daily_df, ["SM", "Spread Margin"])
if sm_col is None:
    daily_df["__SM"] = daily_df[d_pts] - daily_df[d_opp_pts]
    sm_col = "__SM"

tot_col = find_col(daily_df, ["Total Points", "TOTAL_POINTS", "Total"])
if tot_col is None:
    daily_df["__TOTAL"] = daily_df[d_pts] + daily_df[d_opp_pts]
    tot_col = "__TOTAL"

# Date
d_date = find_col(daily_df, ["Date", "Game Date", "Game_Date"])
daily_df["__Date"] = parse_mdy(daily_df[d_date]) if d_date else pd.to_datetime("2000-01-01")

# Moneyline win label
y_win = (daily_df[d_pts] > daily_df[d_opp_pts]).astype(int)

# Spread cover label: SM + Line > 0 (team covers its listed line)
y_cover = pd.Series(np.nan, index=daily_df.index)
if d_line is not None:
    daily_df[d_line] = pd.to_numeric(daily_df[d_line], errors="coerce")
    y_cover = ((daily_df[sm_col] + daily_df[d_line]) > 0).astype(int)

# Over label: TOTAL > OU line
y_over = pd.Series(np.nan, index=daily_df.index)
if d_ou is not None:
    daily_df[d_ou] = pd.to_numeric(daily_df[d_ou], errors="coerce")
    y_over = (daily_df[tot_col] > daily_df[d_ou]).astype(int)

# Coerce numeric features (for training)
for c in daily_numeric_cols:
    daily_df[c] = pd.to_numeric(daily_df[c], errors="coerce")

# categorical columns
cat_candidates = [
    d_team, d_opp,
    "Coach Name","Coach","Coach_Name",
    "Opponent Coach","Opp Coach",
    "Conference","Opponent Conference","Opp Conference",
    "HAN","Home/Away","HomeAway","Location","Loc"
]
cat_cols = [c for c in cat_candidates if c and c in daily_df.columns]
cat_cols = list(dict.fromkeys(cat_cols))

# IMPORTANT: avoid double-assigning a column to num & cat
daily_numeric_cols = [c for c in daily_numeric_cols if c not in cat_cols]

mask_points = daily_df[d_pts].notna() & daily_df[d_opp_pts].notna()
df_train = daily_df.loc[mask_points].copy()

# TEAM PROFILES (median per-team across all numeric features)
team_profiles = df_train.groupby(d_team)[daily_numeric_cols].median(numeric_only=True)
global_medians = df_train[daily_numeric_cols].median(numeric_only=True)

def opp_base_col(col: str) -> Optional[str]:
    if col.upper().startswith("OPP_"):
        return col[4:]
    return None

def get_team_profile(team: str) -> pd.Series:
    if team in team_profiles.index:
        return team_profiles.loc[team]
    return global_medians

# RECENCY WEIGHTS
current_year = int(df_train["__Date"].dt.year.max()) if df_train["__Date"].notna().any() else 2000
sample_weight = np.ones(len(df_train), dtype=float)
recent_mask = (df_train["__Date"].dt.year == current_year)
sample_weight[recent_mask.values] = 3.0

# PREPROCESSOR
num_transformer = SimpleImputer(strategy="median")
try:
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

preproc = ColumnTransformer(
    transformers=[
        ("num", num_transformer, daily_numeric_cols),
        ("cat", cat_transformer, cat_cols)
    ],
    remainder="drop",
    sparse_threshold=0.0
)

X_all = preproc.fit_transform(df_train[daily_numeric_cols + cat_cols])

# Targets aligned to df_train
Y_points = df_train[[d_pts, d_opp_pts]].values
y_win_train = y_win.loc[df_train.index].values.astype(int)

y_cover_train = np.full(len(df_train), np.nan, dtype=float)
if d_line is not None:
    mask_cover_avail = df_train[d_line].notna() & df_train[sm_col].notna()
    y_cover_train[mask_cover_avail.values] = (
        (df_train.loc[mask_cover_avail, sm_col] + df_train.loc[mask_cover_avail, d_line]) > 0
    ).astype(int).values

y_over_train = np.full(len(df_train), np.nan, dtype=float)
if d_ou is not None:
    mask_ou_avail = df_train[d_ou].notna() & df_train[tot_col].notna()
    y_over_train[mask_ou_avail.values] = (
        df_train.loc[mask_ou_avail, tot_col] > df_train.loc[mask_ou_avail, d_ou]
    ).astype(int).values

# =========================================================
# TRAIN MODELS
# =========================================================
st.markdown("### Model Training")
st.caption("Regression for Points/Opp Points + calibrated classifiers for Win, Spread, and Total.")

rf_points = RandomForestRegressor(
    n_estimators=900,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=1,
    max_features="sqrt"
)
rf_points.fit(X_all, Y_points, sample_weight=sample_weight)

def safe_cv_auc(X: np.ndarray, y: np.ndarray, w: np.ndarray, splits: int = 5) -> Optional[float]:
    mask = ~np.isnan(y)
    y_ = y[mask].astype(int)
    X_ = X[mask]
    w_ = w[mask]
    if X_.shape[0] < 10 or len(np.unique(y_)) < 2:
        return None
    class_counts = np.bincount(y_)
    min_class = int(class_counts.min()) if len(class_counts) > 0 else 0
    n_splits = min(splits, X_.shape[0], min_class)
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr, te in skf.split(X_, y_):
        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        )
        clf.fit(X_[tr], y_[tr], sample_weight=w_[tr])
        proba = clf.predict_proba(X_[te])[:, 1]
        scores.append(roc_auc_score(y_[te], proba))
    return float(np.mean(scores)) if scores else None

def safe_cv_acc(X: np.ndarray, y: np.ndarray, w: np.ndarray, splits: int = 5) -> Optional[float]:
    mask = ~np.isnan(y)
    y_ = y[mask].astype(int)
    X_ = X[mask]
    w_ = w[mask]
    if X_.shape[0] < 10 or len(np.unique(y_)) < 2:
        return None
    class_counts = np.bincount(y_)
    min_class = int(class_counts.min()) if len(class_counts) > 0 else 0
    n_splits = min(splits, X_.shape[0], min_class)
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X_, y_):
        clf = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1
        )
        clf.fit(X_[tr], y_[tr], sample_weight=w_[tr])
        pred = clf.predict(X_[te])
        accs.append(accuracy_score(y_[te], pred))
    return float(np.mean(accs)) if accs else None

auc_win   = safe_cv_auc(X_all, y_win_train.astype(float), sample_weight)
auc_cover = safe_cv_auc(X_all, y_cover_train, sample_weight)
auc_over  = safe_cv_auc(X_all, y_over_train, sample_weight)

acc_win   = safe_cv_acc(X_all, y_win_train.astype(float), sample_weight)
acc_cover = safe_cv_acc(X_all, y_cover_train, sample_weight)
acc_over  = safe_cv_acc(X_all, y_over_train, sample_weight)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.write(f"Moneyline/Win AUC: {auc_win:.3f}" if auc_win is not None else "Moneyline/Win AUC: n/a")
    st.write(f"Moneyline/Win ACC: {acc_win:.3f}" if acc_win is not None else "Moneyline/Win ACC: n/a")
with col_m2:
    st.write(f"Spread AUC: {auc_cover:.3f}" if auc_cover is not None else "Spread AUC: n/a")
    st.write(f"Spread ACC: {acc_cover:.3f}" if acc_cover is not None else "Spread ACC: n/a")
with col_m3:
    st.write(f"Total (OU) AUC: {auc_over:.3f}" if auc_over is not None else "Total (OU) AUC: n/a")
    st.write(f"Total (OU) ACC: {acc_over:.3f}" if acc_over is not None else "Total (OU) ACC: n/a")

valid_aucs = [x for x in [auc_win, auc_cover, auc_over] if x is not None]
avg_auc = float(np.mean(valid_aucs)) if valid_aucs else None
valid_accs = [x for x in [acc_win, acc_cover, acc_over] if x is not None]
avg_acc = float(np.mean(valid_accs)) if valid_accs else None

cA, cB = st.columns(2)
with cA:
    st.metric("Average AUC (Win+Spread+OU)", f"{avg_auc:.3f}" if avg_auc is not None else "n/a")
with cB:
    st.metric("Average ACC (Win+Spread+OU)", f"{avg_acc:.3f}" if avg_acc is not None else "n/a")

# Calibrated classifiers (probabilities behave better)
base_win = RandomForestClassifier(n_estimators=800, random_state=42, class_weight="balanced_subsample", n_jobs=-1)
clf_win = CalibratedClassifierCV(base_win, method="isotonic", cv=5)
clf_win.fit(X_all, y_win_train, sample_weight=sample_weight)

clf_cover = None
mask_cover_fit = ~np.isnan(y_cover_train)
if mask_cover_fit.any():
    base_cover = RandomForestClassifier(n_estimators=700, random_state=42, class_weight="balanced_subsample", n_jobs=-1)
    clf_cover = CalibratedClassifierCV(base_cover, method="isotonic", cv=5)
    clf_cover.fit(X_all[mask_cover_fit], y_cover_train[mask_cover_fit].astype(int), sample_weight=sample_weight[mask_cover_fit])

clf_over = None
mask_over_fit = ~np.isnan(y_over_train)
if mask_over_fit.any():
    base_over = RandomForestClassifier(n_estimators=700, random_state=42, class_weight="balanced_subsample", n_jobs=-1)
    clf_over = CalibratedClassifierCV(base_over, method="isotonic", cv=5)
    clf_over.fit(X_all[mask_over_fit], y_over_train[mask_over_fit].astype(int), sample_weight=sample_weight[mask_over_fit])

# =========================================================
# USER CONTROLS (per your goals)
# =========================================================
st.markdown("---")
st.markdown("### Bankroll & Risk Controls")

c1, c2, c3 = st.columns(3)
with c1:
    bet_amount = st.number_input("Bet Amount ($) for today", min_value=1.0, value=100.0, step=5.0)

with c2:
    goal_pct = st.slider("Goal: % increase (higher = riskier)", 5, 50, 25, 1)
    st.caption(f"Target profit goal ≈ ${bet_amount * (goal_pct/100.0):.2f}")

with c3:
    ml_min, ml_max = st.slider("Acceptable ML odds range", -1000, 1000, (-500, 800), 10)

# Thresholds controlled by goal%
min_prob_binary = float(np.interp(goal_pct, [5, 50], [0.70, 0.52]))   # spread/OU must be at least this
edge_floor      = float(np.interp(goal_pct, [5, 50], [0.040, 0.006])) # +EV threshold
min_bets, max_bets = st.slider("Bets minimum / maximum", 1, 100, (10, 100))

# Goal% -> ranking blend (0=safe, 1=chase payout)
alpha = float(np.interp(goal_pct, [5, 50], [0.0, 1.0]))

# =========================================================
# SCHEDULE → FEATURE using team/opponent profiles
# =========================================================
def schedule_row_to_feature(row: pd.Series) -> pd.DataFrame:
    t = str(row[team_col]).strip()
    o = str(row[opp_col]).strip()
    team_prof = get_team_profile(t)
    opp_prof  = get_team_profile(o)

    data: Dict[str, Any] = {}
    for c in daily_numeric_cols:
        val = pd.to_numeric(row.get(c, np.nan), errors="coerce")
        if pd.notna(val):
            data[c] = float(val)
            continue

        base = opp_base_col(c)
        if base is not None:
            data[c] = float(opp_prof.get(base, global_medians.get(c, np.nan)))
        else:
            data[c] = float(team_prof.get(c, global_medians.get(c, np.nan)))

    mapping = {
        d_team: team_col, d_opp: opp_col,
        "Coach Name": coach_col, "Coach": coach_col, "Coach_Name": coach_col,
        "Opponent Coach": opp_coach_col, "Opp Coach": opp_coach_col,
        "Conference": conf_col, "Opponent Conference": opp_conf_col, "Opp Conference": opp_conf_col,
        "HAN": han_col, "Home/Away": han_col, "HomeAway": han_col, "Location": han_col, "Loc": han_col
    }
    for c in cat_cols:
        src = mapping.get(c, None)
        data[c] = str(row.get(src)) if (src and src in row.index) else ""

    one = pd.DataFrame([data], columns=list(dict.fromkeys(daily_numeric_cols + cat_cols)))
    return one

# =========================================================
# SCORE GAMES -> BET CANDIDATES (MULTIPLE PER GAME)
# =========================================================
bet_rows = []
game_rows = []

for _, r in schedule_df.iterrows():
    t = str(r[team_col]).strip()
    o = str(r[opp_col]).strip()
    game_label = f"{t} vs {o}"

    one = schedule_row_to_feature(r)
    X = preproc.transform(one)

    pts_pred = rf_points.predict(X)[0]
    pts, oppp = float(pts_pred[0]), float(pts_pred[1])

    jitter = uniq_jitter(f"{t}|{o}|{r['__Date']}")
    pts  = max(0.0, pts  + jitter)
    oppp = max(0.0, oppp - jitter)

    margin = pts - oppp
    total  = pts + oppp

    line = safe_num(r.get(line_col), default=np.nan) if line_col else np.nan
    ouv  = safe_num(r.get(ou_col),   default=np.nan) if ou_col   else np.nan
    mline= safe_num(r.get(ml_col),   default=np.nan) if ml_col   else np.nan

    is_top25 = truthy_flag(r.get(top25_col)) if (top25_col and top25_col in schedule_df.columns) else False
    is_mm    = truthy_flag(r.get(mm_col))    if (mm_col and mm_col in schedule_df.columns) else False

    conf = r.get(opp_conf_col) if (opp_conf_col and opp_conf_col in r.index) else r.get(conf_col)
    cc = str(conf).strip().upper() if conf is not None else ""
    is_p5 = cc in ("SEC","BIG TEN","B1G","BIG 12","ACC","PAC-12","PAC 12","BIG EAST")

    # Classifier probabilities
    win_p = clamp_prob(clf_win.predict_proba(X)[0, 1])

    cover_prob_team = np.nan
    if np.isfinite(line) and clf_cover is not None:
        cover_prob_team = clamp_prob(clf_cover.predict_proba(X)[0, 1])

    over_prob = np.nan
    if np.isfinite(ouv) and clf_over is not None:
        over_prob = clamp_prob(clf_over.predict_proba(X)[0, 1])

    game_rows.append({
        "Date": r["__Date"],
        "Team": t,
        "Opponent": o,
        "Pred_Points": round(pts, 1),
        "Pred_Opp_Points": round(oppp, 1),
        "Pred_Margin": round(margin, 1),
        "Pred_Total": round(total, 1),
        "WinProb": win_p,
        "Line": line if np.isfinite(line) else None,
        "CoverProb_TeamCoversLine": cover_prob_team if np.isfinite(cover_prob_team) else None,
        "ML": mline if np.isfinite(mline) else None,
        "OU_Line": ouv if np.isfinite(ouv) else None,
        "OverProb": over_prob if np.isfinite(over_prob) else None,
        "IsTop25": is_top25,
        "IsMM": is_mm,
        "IsP5": is_p5,
        "RowObj": r
    })

    # ---------------------------
    # Candidate 1: Moneyline
    # ---------------------------
    if np.isfinite(mline) and (ml_min <= mline <= ml_max):
        ml_dec = american_to_decimal(mline)
        if np.isfinite(ml_dec):
            ml_edge = win_p - (1.0 / ml_dec)
            if np.isfinite(ml_edge) and ml_edge > edge_floor:
                bet_rows.append({
                    "Date": r["__Date"],
                    "GAME": game_label,
                    "BET TYPE": "ML",
                    "BET": f"{t} ML",
                    "VALUE_EDGE": float(ml_edge),
                    "ODDS": int(mline),
                    "ODDS_DEC": float(ml_dec),
                    "PROB": float(win_p),
                    "IsTop25": is_top25,
                    "IsMM": is_mm,
                    "IsP5": is_p5
                })

    # ---------------------------
    # Candidate 2: Spread (binary complement + ML consistency)
    # ---------------------------
    if np.isfinite(line) and np.isfinite(cover_prob_team):
        team_spread = f"{t} {('+' if line > 0 else '')}{line}"
        opp_line = -line
        opp_spread = f"{o} {('+' if opp_line > 0 else '')}{opp_line}"

        spread_pick, spread_prob = pick_binary_side(cover_prob_team, team_spread, opp_spread)

        if spread_pick is not None and np.isfinite(spread_prob):
            # Determine win probability and line sign for the PICKED side
            if spread_pick.startswith(t + " "):
                picked_win_prob = win_p
                picked_line = line
            else:
                picked_win_prob = 1.0 - win_p
                picked_line = opp_line

            # Enforce real-world relationship between win and cover
            spread_prob = enforce_ml_spread_consistency(picked_line, picked_win_prob, spread_prob)

            if np.isfinite(spread_prob) and spread_prob >= min_prob_binary:
                spread_dec = 1.909
                spread_edge = spread_prob - (1.0 / spread_dec)
                if np.isfinite(spread_edge) and spread_edge > edge_floor:
                    bet_rows.append({
                        "Date": r["__Date"],
                        "GAME": game_label,
                        "BET TYPE": "SPREAD",
                        "BET": spread_pick,
                        "VALUE_EDGE": float(spread_edge),
                        "ODDS": -110,
                        "ODDS_DEC": float(spread_dec),
                        "PROB": float(spread_prob),
                        "IsTop25": is_top25,
                        "IsMM": is_mm,
                        "IsP5": is_p5
                    })

    # ---------------------------
    # Candidate 3: Over/Under (binary complement)
    # ---------------------------
    if np.isfinite(ouv) and np.isfinite(over_prob):
        ou_pick, ou_prob_pick = pick_binary_side(over_prob, f"Over {ouv}", f"Under {ouv}")

        if ou_pick is not None and np.isfinite(ou_prob_pick) and ou_prob_pick >= min_prob_binary:
            ou_dec = 1.909
            ou_edge = ou_prob_pick - (1.0 / ou_dec)
            if np.isfinite(ou_edge) and ou_edge > edge_floor:
                bet_rows.append({
                    "Date": r["__Date"],
                    "GAME": game_label,
                    "BET TYPE": "OVER UNDER",
                    "BET": ou_pick,
                    "VALUE_EDGE": float(ou_edge),
                    "ODDS": -110,
                    "ODDS_DEC": float(ou_dec),
                    "PROB": float(ou_prob_pick),
                    "IsTop25": is_top25,
                    "IsMM": is_mm,
                    "IsP5": is_p5
                })

pred_games_df = pd.DataFrame(game_rows)
pred_bets_df  = pd.DataFrame(bet_rows)

# Exclude Penn State if desired
pred_games_df = pred_games_df[~pred_games_df["Team"].apply(is_penn_state)].reset_index(drop=True)
pred_bets_df  = pred_bets_df[~pred_bets_df["BET"].astype(str).str.upper().str.contains("PENN STATE|PENN ST")].reset_index(drop=True)

if pred_bets_df.empty:
    st.info("No +EV bets under current settings (edge/prob/odds range).")
    st.stop()

# =========================================================
# GOAL-BASED RANKING (low goal => better odds/safety; high goal => payout)
# =========================================================
pred_bets_df["_prio"] = pred_bets_df.apply(
    lambda rr: (0 if rr["IsTop25"] else 1, 0 if rr["IsMM"] else 1, 0 if rr["IsP5"] else 1, rr["Date"]),
    axis=1
)

pred_bets_df["PROFIT_MULT"] = pred_bets_df["ODDS_DEC"].apply(profit_multiple)
pred_bets_df["EV_PROFIT_$1"] = pred_bets_df.apply(
    lambda rr: expected_profit_per_dollar(float(rr["PROB"]), float(rr["ODDS_DEC"])),
    axis=1
)

# SCORE blend controlled by goal%
# - alpha ~0 (low goal): prioritize clean edge (safer / better odds behavior)
# - alpha ~1 (high goal): prioritize probability-adjusted payout (EV profit per $1)
pred_bets_df["SCORE"] = (1.0 - alpha) * pred_bets_df["VALUE_EDGE"] + alpha * pred_bets_df["EV_PROFIT_$1"]

pred_bets_df = pred_bets_df.sort_values(by=["_prio", "SCORE"], ascending=[True, False]).reset_index(drop=True)

# =========================================================
# BUILD FINAL PLAN (BET-LEVEL)
# =========================================================
st.markdown("---")
st.markdown("### Suggested Plan (Multi-market — can include multiple bets per game)")

max_available = len(pred_bets_df)
N_total = min(max_bets, max(min_bets, max_available))

# Light diversification targets (still score-driven)
ml_df = pred_bets_df[pred_bets_df["BET TYPE"] == "ML"].copy()
sp_df = pred_bets_df[pred_bets_df["BET TYPE"] == "SPREAD"].copy()
ou_df = pred_bets_df[pred_bets_df["BET TYPE"] == "OVER UNDER"].copy()

ml_target = int(math.floor(0.20 * N_total))
sp_target = int(math.floor(0.40 * N_total))
ou_target = int(math.floor(0.40 * N_total))

picked = []

def take(df, k):
    if k <= 0 or df.empty:
        return
    picked.extend(df.head(k).to_dict("records"))

take(sp_df, min(sp_target, len(sp_df)))
take(ou_df, min(ou_target, len(ou_df)))
take(ml_df, min(ml_target, len(ml_df)))

if len(picked) < N_total:
    need = N_total - len(picked)
    already_keys = set((p["Date"], p["GAME"], p["BET TYPE"], p["BET"]) for p in picked)
    remaining = pred_bets_df.copy()
    remaining["_key"] = remaining.apply(lambda rr: (rr["Date"], rr["GAME"], rr["BET TYPE"], rr["BET"]), axis=1)
    remaining = remaining[~remaining["_key"].isin(already_keys)]
    picked.extend(remaining.head(need).to_dict("records"))

plan = pd.DataFrame(picked).copy()
plan = plan.sort_values(by=["_prio", "SCORE"], ascending=[True, False]).reset_index(drop=True)

# Stake sizing in $ based on SCORE+EV strength (never negative)
# Use positive part of SCORE and VALUE_EDGE to avoid weird allocations
w = (plan["SCORE"].astype(float)).clip(lower=1e-6)
total_w = float(w.sum()) if len(w) else 1.0
plan["VALUE"] = (w / total_w) * float(bet_amount)

# round to nearest $1 and min $1
plan["VALUE"] = plan["VALUE"].round(0).clip(lower=1.0)

# Renormalize after rounding to stay close to bet_amount
def renorm_dollars(df, target):
    diff = float(target) - float(df["VALUE"].sum())
    if abs(diff) < 0.5:
        return df
    order = df.sort_values("SCORE", ascending=False).index.tolist()
    step = 1.0 if diff > 0 else -1.0
    i = 0
    while abs(diff) >= 0.5 and i < len(order) * 10:
        idx = order[i % len(order)]
        newv = df.at[idx, "VALUE"] + step
        if newv >= 1.0:
            df.at[idx, "VALUE"] = newv
            diff = float(target) - float(df["VALUE"].sum())
        i += 1
    return df

plan = renorm_dollars(plan, float(bet_amount))

# Export columns (exact)
plan["Probability of accuracy"] = plan["PROB"].astype(float)

# Payout rounded to 2 decimals (FIX #1)
plan["Payout if win"] = plan.apply(
    lambda rr: round(payout_if_win_total(float(rr["VALUE"]), float(rr["ODDS"])), 2),
    axis=1
)

download_df = plan[[
    "GAME",
    "BET TYPE",
    "BET",
    "VALUE",
    "ODDS",
    "Probability of accuracy",
    "Payout if win"
]].copy()

st.dataframe(download_df, use_container_width=True)

st.download_button(
    "Download Betting Plan CSV",
    data=download_df.to_csv(index=False),
    file_name="betting_plan.csv",
    mime="text/csv"
)

# =========================================================
# DRILLDOWN — optional (kept)
# =========================================================
st.markdown("---")
st.markdown("### Drilldown (matchup details)")

def find_opponent_row_raw(team: str, opp: str, date_val: pd.Timestamp) -> Optional[pd.Series]:
    if team_col not in schedule_df_raw.columns or opp_col not in schedule_df_raw.columns:
        return None
    df = schedule_df_raw.copy()
    df["__Date"] = parse_mdy(df[date_col])
    mask = (
        df[team_col].astype(str).str.strip() == opp
    ) & (
        df[opp_col].astype(str).str.strip() == team
    ) & (
        df["__Date"].dt.date == date_val.date()
    )
    res = df.loc[mask]
    return res.iloc[0] if not res.empty else None

for _, rr in pred_games_df.iterrows():
    label = (
        f"{rr['Date'].strftime('%b %d, %Y')} — "
        f"{rr['Team']} vs {rr['Opponent']} — "
        f"Pred {int(round(rr['Pred_Points']))}-{int(round(rr['Pred_Opp_Points']))}"
    )
    with st.expander(label, expanded=False):
        row_team = rr["RowObj"]
        row_opp  = find_opponent_row_raw(rr["Team"], rr["Opponent"], rr["Date"])

        t_conf = row_team.get(conf_col) if (conf_col and conf_col in row_team.index) else ""
        o_conf = (
            row_team.get(opp_conf_col) if (opp_conf_col and opp_conf_col in row_team.index) else
            (row_opp.get(conf_col) if (row_opp is not None and conf_col in row_opp.index) else "")
        )

        t_coach = row_team.get(coach_col) if (coach_col and coach_col in row_team.index) else ""
        o_coach = (
            row_team.get(opp_coach_col) if (opp_coach_col and opp_coach_col in row_team.index) else
            (row_opp.get(coach_col) if (row_opp is not None and coach_col in row_opp.index) else "")
        )

        cL, cR = st.columns(2)
        with cL:
            st.write(f"**{rr['Team']}**")
            st.write(f"Coach: {t_coach}")
            st.write(f"Conference: {t_conf}")
        with cR:
            st.write(f"**{rr['Opponent']}**")
            st.write(f"Coach: {o_coach}")
            st.write(f"Conference: {o_conf}")

        st.markdown("---")
        st.write("Top-50 Rank Categories (≤ 50)")

        left_top50 = extract_top50_any(row_team)
        right_top50 = extract_top50_any(row_opp)

        cL2, cR2 = st.columns(2)
        with cL2:
            st.write(f"**{rr['Team']}**")
            st.write("None in top 50" if not left_top50 else "")
            if left_top50:
                st.dataframe(pd.DataFrame([{"Category": k, "Rank": v} for k, v in left_top50]),
                             use_container_width=True)
        with cR2:
            st.write(f"**{rr['Opponent']}**")
            st.write("None in top 50" if not right_top50 else "")
            if right_top50:
                st.dataframe(pd.DataFrame([{"Category": k, "Rank": v} for k, v in right_top50]),
                             use_container_width=True)
                
