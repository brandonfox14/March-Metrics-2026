# APP/Pages/6_Betting.py
# =========================================================
# Betting Page — multi-market plan (ML/Spread/OU)
# (Recency-weighted learning + improved betting logic)
# =========================================================

import os
import math
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
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

def parse_dates_mixed(series_like: pd.Series) -> pd.Series:
    """
    Handles:
      - November 28, 2025
      - 12/8/2025
      - 7-Nov-24
      - and most other common formats
    """
    s = series_like.astype(str).str.strip()
    out = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)

    # Pass 2: try explicit common formats for stubborn entries
    mask = out.isna()
    if mask.any():
        s2 = s[mask]
        for fmt in ("%m/%d/%Y", "%m/%d/%y", "%d-%b-%y", "%d-%b-%Y", "%b %d, %Y", "%B %d, %Y"):
            tmp = pd.to_datetime(s2, errors="coerce", format=fmt)
            out.loc[mask] = out.loc[mask].fillna(tmp)
            mask = out.isna()
            if not mask.any():
                break

    return out

def safe_num(x, default=np.nan) -> float:
    try:
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default

def clamp_prob(p: float, lo: float = 0.001, hi: float = 0.999) -> float:
    if not np.isfinite(p):
        return np.nan
    return float(min(hi, max(lo, p)))

def american_to_decimal(odds: float) -> float:
    o = safe_num(odds, np.nan)
    if not np.isfinite(o) or o == 0:
        return np.nan
    if o > 0:
        return 1.0 + (o / 100.0)
    return 1.0 + (100.0 / abs(o))

def implied_prob_american(odds: float) -> float:
    """
    Standard implied probability ignoring vig:
      +100 -> 0.50
      -130 -> 130/(130+100)=0.5652
    """
    o = safe_num(odds, np.nan)
    if not np.isfinite(o) or o == 0:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)

def expected_profit_per_dollar(prob: float, dec: float) -> float:
    # EV in profit units per $1 stake
    if not (np.isfinite(prob) and np.isfinite(dec) and dec > 1.0):
        return np.nan
    p = clamp_prob(prob)
    return float(p * (dec - 1.0) - (1.0 - p))

def payout_if_win_total(stake: float, odds_us: float) -> float:
    # Total return if win = stake * decimal_odds
    if not np.isfinite(stake) or stake <= 0:
        return np.nan
    dec = american_to_decimal(odds_us)
    if not np.isfinite(dec):
        return np.nan
    return float(stake * dec)

def round_to_tenth(x: float) -> float:
    return float(np.round(x * 10.0) / 10.0)

def units_from_edge(prob: float, odds_us: float, max_units: float = 5.0) -> float:
    """
    Fractional Kelly-like sizing, converted to "units".
    Enforces:
      - > 0.5
      - increments of 0.1
    """
    dec = american_to_decimal(odds_us)
    if not (np.isfinite(prob) and np.isfinite(dec) and dec > 1.0):
        return np.nan

    # Kelly fraction: f* = (p*dec - 1)/(dec - 1)
    p = clamp_prob(prob)
    f = (p * dec - 1.0) / (dec - 1.0)

    # Conservative fractional Kelly (cut it down hard)
    f = max(0.0, min(f, 0.10))  # cap at 10% bankroll
    units = 0.5 + (f / 0.10) * (max_units - 0.5)  # map [0,0.10] -> [0.5,max_units]
    units = max(0.51, units)
    return max(0.6, round_to_tenth(units))

def truthy_flag(v) -> bool:
    if pd.isna(v):
        return False
    s = str(v).strip().upper()
    return s not in ("0", "", "N", "NO", "FALSE", "F", "NA", "NAN", "NONE")

def enforce_ml_spread_consistency(line: float, win_prob: float, cover_prob: float) -> float:
    """
    If line < 0 (favorite): P(cover) <= P(win)
    If line > 0 (dog):      P(cover) >= P(win)
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

def pick_binary_side(p_event: float, label_event: str, label_other: str) -> Tuple[Optional[str], float]:
    p_event = clamp_prob(p_event)
    if not np.isfinite(p_event):
        return None, np.nan
    p_other = 1.0 - p_event
    if p_event >= 0.5:
        return label_event, p_event
    return label_other, p_other

# =========================================================
# LOAD DATA
# =========================================================
schedule_df = load_csv(SCHEDULE_FILE)
daily_df    = load_csv(DAILY_FILE)
if schedule_df is None or daily_df is None:
    st.stop()

# =========================================================
# COLUMN DETECTION
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

schedule_df["__Date"] = parse_dates_mixed(schedule_df[date_col])
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

# DAILY columns
d_team = find_col(daily_df, ["Team","Teams","team"])
d_opp  = find_col(daily_df, ["Opponent","Opp","opponent"])
d_pts  = find_col(daily_df, ["Points"])
d_opp_pts = find_col(daily_df, ["Opp Points","Opp_Points","OppPoints"])
d_line = find_col(daily_df, ["Line"])
d_ou   = find_col(daily_df, ["Over/Under Line","OverUnder","Over Under Line","O/U"])
d_ml   = find_col(daily_df, ["ML","Moneyline","Money Line"])
d_date = find_col(daily_df, ["Date", "Game Date", "Game_Date"])

if d_team is None or d_opp is None or d_pts is None or d_opp_pts is None:
    st.error("Daily predictor must include Team, Opponent, Points, Opp Points.")
    st.stop()

# Coerce key numerics
daily_df[d_pts]     = pd.to_numeric(daily_df[d_pts], errors="coerce")
daily_df[d_opp_pts] = pd.to_numeric(daily_df[d_opp_pts], errors="coerce")
if d_line and d_line in daily_df.columns:
    daily_df[d_line] = pd.to_numeric(daily_df[d_line], errors="coerce")
if d_ou and d_ou in daily_df.columns:
    daily_df[d_ou] = pd.to_numeric(daily_df[d_ou], errors="coerce")
if d_ml and d_ml in daily_df.columns:
    daily_df[d_ml] = pd.to_numeric(daily_df[d_ml], errors="coerce")

daily_df["__Date"] = parse_dates_mixed(daily_df[d_date]) if d_date else pd.to_datetime("2000-01-01")
daily_df = daily_df.dropna(subset=["__Date"]).copy()

# Targets
daily_df["__SM"] = daily_df[d_pts] - daily_df[d_opp_pts]
daily_df["__TOTAL"] = daily_df[d_pts] + daily_df[d_opp_pts]

y_win = (daily_df[d_pts] > daily_df[d_opp_pts]).astype(int)

y_cover = np.full(len(daily_df), np.nan, dtype=float)
if d_line:
    mask_cov = daily_df[d_line].notna() & daily_df["__SM"].notna()
    y_cover[mask_cov.values] = ((daily_df.loc[mask_cov, "__SM"] + daily_df.loc[mask_cov, d_line]) > 0).astype(int).values

y_over = np.full(len(daily_df), np.nan, dtype=float)
if d_ou:
    mask_ou = daily_df[d_ou].notna() & daily_df["__TOTAL"].notna()
    y_over[mask_ou.values] = (daily_df.loc[mask_ou, "__TOTAL"] > daily_df.loc[mask_ou, d_ou]).astype(int).values

# Feature columns: numeric + categorical (keep your “use all 214” intent)
daily_numeric_cols = daily_df.select_dtypes(include=[np.number]).columns.tolist()

# remove targets from features
for drop_c in [d_pts, d_opp_pts, "__SM", "__TOTAL"]:
    if drop_c in daily_numeric_cols:
        daily_numeric_cols.remove(drop_c)

cat_candidates = [
    d_team, d_opp,
    "Coach Name","Coach","Coach_Name",
    "Opponent Coach","Opp Coach",
    "Conference","Opponent Conference","Opp Conference",
    "HAN","Home/Away","HomeAway","Location","Loc"
]
cat_cols = [c for c in cat_candidates if c and c in daily_df.columns]
cat_cols = list(dict.fromkeys(cat_cols))
daily_numeric_cols = [c for c in daily_numeric_cols if c not in cat_cols]

# numeric coercion
for c in daily_numeric_cols:
    daily_df[c] = pd.to_numeric(daily_df[c], errors="coerce")

mask_points = daily_df[d_pts].notna() & daily_df[d_opp_pts].notna()
df_train = daily_df.loc[mask_points].copy()

# TEAM PROFILES (median per-team across numeric features)
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

# =========================================================
# RECENCY WEIGHTS (exponential decay)
# =========================================================
st.markdown("### Model Training")
st.caption("Recency-weighted: recent games matter more; older games still provide baseline.")

max_date = df_train["__Date"].max()
half_life_days = st.slider("Recency half-life (days)", 7, 120, 35, 1,
                           help="Smaller = recent games dominate more. Larger = flatter weighting.")
days_since = (max_date - df_train["__Date"]).dt.days.clip(lower=0).astype(float)
# weight = 0.5^(days_since/half_life)
sample_weight = np.power(0.5, days_since / float(half_life_days)).values
# keep a small floor so old games still count
sample_weight = np.clip(sample_weight, 0.10, 1.0)

# =========================================================
# PREPROCESSOR
# =========================================================
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

Y_points = df_train[[d_pts, d_opp_pts]].values.astype(float)
y_win_train = y_win.loc[df_train.index].values.astype(int)

y_cover_train = np.full(len(df_train), np.nan, dtype=float)
if d_line:
    mask_cov = df_train[d_line].notna()
    y_cover_train[mask_cov.values] = ((df_train.loc[mask_cov, "__SM"] + df_train.loc[mask_cov, d_line]) > 0).astype(int).values

y_over_train = np.full(len(df_train), np.nan, dtype=float)
if d_ou:
    mask_ou = df_train[d_ou].notna()
    y_over_train[mask_ou.values] = (df_train.loc[mask_ou, "__TOTAL"] > df_train.loc[mask_ou, d_ou]).astype(int).values

# =========================================================
# SAFE CV METRICS
# =========================================================
def safe_cv_auc(X: np.ndarray, y: np.ndarray, w: np.ndarray, splits: int = 5) -> Optional[float]:
    mask = ~np.isnan(y)
    y_ = y[mask].astype(int)
    X_ = X[mask]
    w_ = w[mask]
    if X_.shape[0] < 20 or len(np.unique(y_)) < 2:
        return None
    class_counts = np.bincount(y_)
    min_class = int(class_counts.min()) if len(class_counts) else 0
    n_splits = max(2, min(splits, X_.shape[0], min_class))
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr, te in skf.split(X_, y_):
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
            max_features="sqrt"
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
    if X_.shape[0] < 20 or len(np.unique(y_)) < 2:
        return None
    class_counts = np.bincount(y_)
    min_class = int(class_counts.min()) if len(class_counts) else 0
    n_splits = max(2, min(splits, X_.shape[0], min_class))
    if n_splits < 2:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(X_, y_):
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
            max_features="sqrt"
        )
        clf.fit(X_[tr], y_[tr], sample_weight=w_[tr])
        pred = clf.predict(X_[te])
        accs.append(accuracy_score(y_[te], pred))
    return float(np.mean(accs)) if accs else None

# =========================================================
# MODEL BUNDLE (single object producing probabilities)
# =========================================================
@dataclass
class ModelBundle:
    preproc: ColumnTransformer
    rf_points: RandomForestRegressor
    clf_win: CalibratedClassifierCV
    clf_cover: Optional[CalibratedClassifierCV]
    clf_over: Optional[CalibratedClassifierCV]

@st.cache_resource(show_spinner=True)
def train_bundle(X_all: np.ndarray,
                 Y_points: np.ndarray,
                 y_win_train: np.ndarray,
                 y_cover_train: np.ndarray,
                 y_over_train: np.ndarray,
                 sample_weight: np.ndarray,
                 preproc: ColumnTransformer) -> Tuple[ModelBundle, Dict[str, Optional[float]]]:

    rf_points = RandomForestRegressor(
        n_estimators=700,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1,
        max_features="sqrt"
    )
    rf_points.fit(X_all, Y_points, sample_weight=sample_weight)

    # Calibrated win classifier
    base_win = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
        max_features="sqrt"
    )
    clf_win = CalibratedClassifierCV(base_win, method="isotonic", cv=5)
    clf_win.fit(X_all, y_win_train, sample_weight=sample_weight)

    clf_cover = None
    mask_cover = ~np.isnan(y_cover_train)
    if mask_cover.any():
        base_cover = RandomForestClassifier(
            n_estimators=550,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
            max_features="sqrt"
        )
        clf_cover = CalibratedClassifierCV(base_cover, method="isotonic", cv=5)
        clf_cover.fit(X_all[mask_cover], y_cover_train[mask_cover].astype(int), sample_weight=sample_weight[mask_cover])

    clf_over = None
    mask_over = ~np.isnan(y_over_train)
    if mask_over.any():
        base_over = RandomForestClassifier(
            n_estimators=550,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
            max_features="sqrt"
        )
        clf_over = CalibratedClassifierCV(base_over, method="isotonic", cv=5)
        clf_over.fit(X_all[mask_over], y_over_train[mask_over].astype(int), sample_weight=sample_weight[mask_over])

    metrics = {
        "auc_win":   safe_cv_auc(X_all, y_win_train.astype(float), sample_weight),
        "auc_cov":   safe_cv_auc(X_all, y_cover_train, sample_weight),
        "auc_over":  safe_cv_auc(X_all, y_over_train, sample_weight),
        "acc_win":   safe_cv_acc(X_all, y_win_train.astype(float), sample_weight),
        "acc_cov":   safe_cv_acc(X_all, y_cover_train, sample_weight),
        "acc_over":  safe_cv_acc(X_all, y_over_train, sample_weight),
    }

    bundle = ModelBundle(
        preproc=preproc,
        rf_points=rf_points,
        clf_win=clf_win,
        clf_cover=clf_cover,
        clf_over=clf_over
    )
    return bundle, metrics

bundle, metrics = train_bundle(X_all, Y_points, y_win_train, y_cover_train, y_over_train, sample_weight, preproc)

m1, m2, m3 = st.columns(3)
with m1:
    st.write(f"Win AUC: {metrics['auc_win']:.3f}" if metrics["auc_win"] is not None else "Win AUC: n/a")
    st.write(f"Win ACC: {metrics['acc_win']:.3f}" if metrics["acc_win"] is not None else "Win ACC: n/a")
with m2:
    st.write(f"Spread AUC: {metrics['auc_cov']:.3f}" if metrics["auc_cov"] is not None else "Spread AUC: n/a")
    st.write(f"Spread ACC: {metrics['acc_cov']:.3f}" if metrics["acc_cov"] is not None else "Spread ACC: n/a")
with m3:
    st.write(f"OU AUC: {metrics['auc_over']:.3f}" if metrics["auc_over"] is not None else "OU AUC: n/a")
    st.write(f"OU ACC: {metrics['acc_over']:.3f}" if metrics["acc_over"] is not None else "OU ACC: n/a")

# =========================================================
# USER CONTROLS (hard rules you specified)
# =========================================================
st.markdown("---")
st.markdown("### Bankroll & Slate Controls")

c1, c2, c3, c4 = st.columns(4)
with c1:
    bankroll_dollars = st.number_input("Total $ to allocate today", min_value=10.0, value=200.0, step=10.0)
with c2:
    N_min, N_max = st.slider("Bets min / max", 1, 60, (8, 20))
with c3:
    max_units = st.slider("Max units per bet", 1.0, 10.0, 5.0, 0.5)
with c4:
    max_type_pct = st.slider("Max % of slate per bet type", 20, 60, 45, 1)

st.caption("Hard thresholds (per your rules): ML requires 10% edge vs implied; Spread/OU require 65%+ probability. "
           "If |spread| ≤ 3.5, choose either ML or Spread (whichever has higher EV), not both.")

ML_EDGE_MIN = 0.10
PROB_MIN_SPREAD = 0.65
PROB_MIN_OU = 0.65

# =========================================================
# SCHEDULE → FEATURE using team/opponent profiles
# =========================================================
def schedule_row_to_feature(row: pd.Series) -> pd.DataFrame:
    t = str(row[team_col]).strip()
    o = str(row[opp_col]).strip()
    team_prof = get_team_profile(t)
    opp_prof  = get_team_profile(o)

    data: Dict[str, Any] = {}

    # numeric features: use schedule value if present; else profile-based fill
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

    # categorical mapping
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

    return pd.DataFrame([data], columns=list(dict.fromkeys(daily_numeric_cols + cat_cols)))

# =========================================================
# PREDICTION FORMAT RULES (whole numbers, no ties)
# =========================================================
def finalize_score_pair(team_raw: float, opp_raw: float, key: str) -> Tuple[int, int]:
    # whole numbers
    t = int(np.round(max(0.0, team_raw)))
    o = int(np.round(max(0.0, opp_raw)))

    # no ties
    if t == o:
        # break tie deterministically using hash key and raw margin
        if team_raw > opp_raw:
            t += 1
        elif opp_raw > team_raw:
            o += 1
        else:
            # truly identical -> deterministic bump
            if (abs(hash(key)) % 2) == 0:
                t += 1
            else:
                o += 1
    return t, o

# =========================================================
# BUILD BET CANDIDATES PER GAME
# =========================================================
bet_rows: List[Dict[str, Any]] = []
game_rows: List[Dict[str, Any]] = []

for _, r in schedule_df.iterrows():
    t = str(r[team_col]).strip()
    o = str(r[opp_col]).strip()
    game_label = f"{t} vs {o}"
    date_val = r["__Date"]

    one = schedule_row_to_feature(r)
    X = bundle.preproc.transform(one)

    # points prediction
    pts_pred = bundle.rf_points.predict(X)[0]
    pts_raw, opp_raw = float(pts_pred[0]), float(pts_pred[1])
    pts_i, opp_i = finalize_score_pair(pts_raw, opp_raw, f"{t}|{o}|{date_val}")

    margin = pts_i - opp_i
    total = pts_i + opp_i

    # schedule lines/odds
    line = safe_num(r.get(line_col), np.nan) if line_col else np.nan
    ouv  = safe_num(r.get(ou_col),   np.nan) if ou_col   else np.nan
    ml   = safe_num(r.get(ml_col),   np.nan) if ml_col   else np.nan

    is_top25 = truthy_flag(r.get(top25_col)) if top25_col else False
    is_mm    = truthy_flag(r.get(mm_col)) if mm_col else False

    conf = r.get(opp_conf_col) if opp_conf_col else r.get(conf_col)
    cc = str(conf).strip().upper() if conf is not None else ""
    is_p5 = cc in ("SEC","BIG TEN","B1G","BIG 12","ACC","PAC-12","PAC 12","BIG EAST")

    # probabilities
    win_p_team = clamp_prob(bundle.clf_win.predict_proba(X)[0, 1])  # P(team wins)

    cover_p_team = np.nan
    if np.isfinite(line) and bundle.clf_cover is not None:
        cover_p_team = clamp_prob(bundle.clf_cover.predict_proba(X)[0, 1])  # P(team covers its listed line

    over_p = np.nan
    if np.isfinite(ouv) and bundle.clf_over is not None:
        over_p = clamp_prob(bundle.clf_over.predict_proba(X)[0, 1])  # P(over hits)

    game_rows.append({
        "Date": date_val,
        "Team": t,
        "Opponent": o,
        "Pred Points": pts_i,
        "Pred Opp Points": opp_i,
        "Pred Margin": margin,
        "Pred Total": total,
        "WinProb(Team)": win_p_team,
        "Line": line,
        "ML": ml,
        "OU": ouv
    })

    # -----------------------
    # Candidate: MONEYLINE
    # -----------------------
    ml_candidates = []
    if np.isfinite(ml):
        # bet on TEAM
        team_imp = implied_prob_american(ml)
        team_ev = expected_profit_per_dollar(win_p_team, american_to_decimal(ml))
        if np.isfinite(team_imp) and np.isfinite(team_ev):
            if win_p_team >= (team_imp + ML_EDGE_MIN):
                ml_candidates.append({
                    "Date": date_val, "GAME": game_label, "BET TYPE": "ML",
                    "BET": f"{t} ML",
                    "ODDS": float(ml),
                    "ODDS_DEC": float(american_to_decimal(ml)),
                    "PROB": float(win_p_team),
                    "IMPLIED_PROB": float(team_imp),
                    "EV_$1": float(team_ev),
                    "IsTop25": is_top25, "IsMM": is_mm, "IsP5": is_p5
                })

        # bet on OPP (opposite ML is not always present; if your file only gives one side, skip)
        # If your schedule has both sides (Team ML and Opp ML columns), extend this block.

    # -----------------------
    # Candidate: SPREAD (odds -110)
    # Logic: if Score + Line > OppScore then cover (for label creation),
    # For prediction: we choose the side with higher cover probability (binary complement).
    # -----------------------
    spread_candidates = []
    if np.isfinite(line) and np.isfinite(cover_p_team):
        # for TEAM, line as given; for OPP, opposite line
        opp_line = -line
        team_spread_label = f"{t} {line:+g}"
        opp_spread_label  = f"{o} {opp_line:+g}"

        spread_pick, spread_prob = pick_binary_side(cover_p_team, team_spread_label, opp_spread_label)

        # determine win prob of picked side for consistency clamp
        if spread_pick is not None:
            if spread_pick.startswith(t + " "):
                picked_win_p = win_p_team
                picked_line = line
            else:
                picked_win_p = 1.0 - win_p_team
                picked_line = opp_line

            spread_prob = enforce_ml_spread_consistency(picked_line, picked_win_p, spread_prob)

            # threshold rule
            if np.isfinite(spread_prob) and spread_prob >= PROB_MIN_SPREAD:
                odds_spread = -110.0
                dec_spread = american_to_decimal(odds_spread)
                imp_spread = 1.0 / dec_spread  # ~0.5238
                ev_spread = expected_profit_per_dollar(spread_prob, dec_spread)
                spread_candidates.append({
                    "Date": date_val, "GAME": game_label, "BET TYPE": "SPREAD",
                    "BET": spread_pick,
                    "ODDS": odds_spread,
                    "ODDS_DEC": float(dec_spread),
                    "PROB": float(spread_prob),
                    "IMPLIED_PROB": float(imp_spread),
                    "EV_$1": float(ev_spread),
                    "IsTop25": is_top25, "IsMM": is_mm, "IsP5": is_p5
                })

    # -----------------------
    # Candidate: OVER/UNDER (odds -110)
    # -----------------------
    ou_candidates = []
    if np.isfinite(ouv) and np.isfinite(over_p):
        ou_pick, ou_prob_pick = pick_binary_side(over_p, f"Over {ouv:g}", f"Under {ouv:g}")
        if ou_pick is not None and np.isfinite(ou_prob_pick) and ou_prob_pick >= PROB_MIN_OU:
            odds_ou = -110.0
            dec_ou = american_to_decimal(odds_ou)
            imp_ou = 1.0 / dec_ou
            ev_ou = expected_profit_per_dollar(ou_prob_pick, dec_ou)
            ou_candidates.append({
                "Date": date_val, "GAME": game_label, "BET TYPE": "OVER UNDER",
                "BET": ou_pick,
                "ODDS": odds_ou,
                "ODDS_DEC": float(dec_ou),
                "PROB": float(ou_prob_pick),
                "IMPLIED_PROB": float(imp_ou),
                "EV_$1": float(ev_ou),
                "IsTop25": is_top25, "IsMM": is_mm, "IsP5": is_p5
            })

    # -----------------------
    # Special rule: If |spread| ≤ 3.5 -> choose best EV between ML and SPREAD (not both)
    # Otherwise: allow both if they pass thresholds
    # -----------------------
    if np.isfinite(line) and abs(line) <= 3.5:
        pool = []
        pool.extend(spread_candidates)
        pool.extend(ml_candidates)
        if pool:
            best = sorted(pool, key=lambda d: (d.get("EV_$1", -1e9), d.get("PROB", 0.0)), reverse=True)[0]
            bet_rows.append(best)
    else:
        bet_rows.extend(ml_candidates)
        bet_rows.extend(spread_candidates)
    bet_rows.extend(ou_candidates)

pred_games_df = pd.DataFrame(game_rows)
pred_bets_df  = pd.DataFrame(bet_rows)

if pred_games_df.empty:
    st.info("No games found in schedule after parsing.")
    st.stop()

st.markdown("---")
st.markdown("### Predictions (Scores)")
st.dataframe(
    pred_games_df.sort_values(["Date","Team"]).reset_index(drop=True),
    use_container_width=True
)

if pred_bets_df.empty:
    st.info("No bets met your thresholds today (ML edge / Spread+OU 65%+).")
    st.stop()

# =========================================================
# SLATE SELECTION (balance types + 45% max per type)
# =========================================================
st.markdown("---")
st.markdown("### Betting Plan")

# Rank candidates by EV primarily, then probability
pred_bets_df["SCORE"] = pred_bets_df["EV_$1"].astype(float)
pred_bets_df = pred_bets_df.sort_values(["SCORE", "PROB"], ascending=[False, False]).reset_index(drop=True)

# Determine target total bets (pick top within [N_min, N_max], but don’t exceed available)
N_total = int(min(max(N_min, min(N_max, len(pred_bets_df))), len(pred_bets_df)))

max_per_type = int(max(1, math.floor((max_type_pct / 100.0) * N_total)))

type_targets = {
    "ML": N_total / 3.0,
    "SPREAD": N_total / 3.0,
    "OVER UNDER": N_total / 3.0
}

type_counts = {"ML": 0, "SPREAD": 0, "OVER UNDER": 0}
picked_rows: List[Dict[str, Any]] = []

def soft_penalty(bet_type: str) -> float:
    # penalty if over target; encourages balance
    over = max(0.0, (type_counts.get(bet_type, 0) + 1) - type_targets.get(bet_type, 0.0))
    return over * 0.02

# Iterate candidates in order, enforce 45% cap + soft balance
for _, rr in pred_bets_df.iterrows():
    if len(picked_rows) >= N_total:
        break
    bt = rr["BET TYPE"]
    if bt not in type_counts:
        continue
    if type_counts[bt] >= max_per_type:
        continue

    # avoid duplicates (same game + bet)
    key = (rr["Date"], rr["GAME"], rr["BET TYPE"], rr["BET"])
    already = set((p["Date"], p["GAME"], p["BET TYPE"], p["BET"]) for p in picked_rows)
    if key in already:
        continue

    # Accept, but bias selection score toward balance
    row = rr.to_dict()
    row["_bal_score"] = float(row["SCORE"]) - soft_penalty(bt)
    picked_rows.append(row)
    type_counts[bt] += 1

# If still short (caps too tight), fill remaining ignoring balance penalty but still respecting caps
if len(picked_rows) < N_total:
    need = N_total - len(picked_rows)
    already = set((p["Date"], p["GAME"], p["BET TYPE"], p["BET"]) for p in picked_rows)

    remain = pred_bets_df.copy()
    remain["_key"] = list(zip(remain["Date"], remain["GAME"], remain["BET TYPE"], remain["BET"]))
    remain = remain[~remain["_key"].isin(already)]

    for _, rr in remain.iterrows():
        if need <= 0:
            break
        bt = rr["BET TYPE"]
        if bt not in type_counts:
            continue
        if type_counts[bt] >= max_per_type:
            continue
        row = rr.to_dict()
        row["_bal_score"] = float(row["SCORE"])
        picked_rows.append(row)
        type_counts[bt] += 1
        need -= 1

plan = pd.DataFrame(picked_rows).copy()
plan = plan.sort_values(["_bal_score", "SCORE", "PROB"], ascending=[False, False, False]).reset_index(drop=True)

# =========================================================
# UNITS + $ ALLOCATION
# =========================================================
plan["UNITS"] = plan.apply(lambda r: units_from_edge(float(r["PROB"]), float(r["ODDS"]), max_units=float(max_units)), axis=1)
plan["UNITS"] = plan["UNITS"].replace([np.inf, -np.inf], np.nan).fillna(0.6)
plan["UNITS"] = plan["UNITS"].clip(lower=0.6)

# Allocate dollars proportional to units
w = plan["UNITS"].astype(float).clip(lower=0.1)
total_w = float(w.sum()) if len(w) else 1.0
plan["VALUE"] = (w / total_w) * float(bankroll_dollars)

# Round VALUE to nearest $1, keep >= $1
plan["VALUE"] = plan["VALUE"].round(0).clip(lower=1.0)

# Renormalize after rounding to match bankroll closely
def renorm_dollars(df: pd.DataFrame, target: float) -> pd.DataFrame:
    diff = float(target) - float(df["VALUE"].sum())
    if abs(diff) < 0.5 or df.empty:
        return df
    order = df.sort_values("SCORE", ascending=False).index.tolist()
    step = 1.0 if diff > 0 else -1.0
    i = 0
    while abs(diff) >= 0.5 and i < len(order) * 20:
        idx = order[i % len(order)]
        newv = df.at[idx, "VALUE"] + step
        if newv >= 1.0:
            df.at[idx, "VALUE"] = newv
            diff = float(target) - float(df["VALUE"].sum())
        i += 1
    return df

plan = renorm_dollars(plan, float(bankroll_dollars))

# Export fields
plan["Probability of accuracy"] = plan["PROB"].astype(float)
plan["Payout if win"] = plan.apply(
    lambda rr: round(payout_if_win_total(float(rr["VALUE"]), float(rr["ODDS"])), 2),
    axis=1
)

download_df = plan[[
    "GAME",
    "BET TYPE",
    "BET",
    "UNITS",
    "VALUE",
    "ODDS",
    "Probability of accuracy",
    "Payout if win"
]].copy()

# Show type mix
mix = download_df["BET TYPE"].value_counts(dropna=False)
st.caption(f"Type mix (cap {max_type_pct}%): " +
           ", ".join([f"{k}={int(v)}" for k, v in mix.items()]))

st.dataframe(download_df, use_container_width=True)

st.download_button(
    "Download Betting Plan CSV",
    data=download_df.to_csv(index=False),
    file_name="betting_plan.csv",
    mime="text/csv"
)
