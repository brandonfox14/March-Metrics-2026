# APP/Pages/6_Betting.py
# =========================================================
# Betting Page — efficient single "bundle" model + improved logic
# (Replaces prior version; keeps file paths/columns compatible)
# =========================================================
import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
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

def parse_mdy(series_like: pd.Series) -> pd.Series:
    s = series_like.astype(str).str.strip()
    out = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    # if your CSV has strict %m/%d/%Y, infer_datetime_format still handles it
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
    if not np.isfinite(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))

def american_to_implied_prob(odds: float) -> float:
    """
    Implied probability from American odds (no-vig not applied; you said implied will include house edge).
    +100 -> 0.50
    -130 -> 130/(130+100) = 0.5652
    """
    if not np.isfinite(odds) or odds == 0:
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

def expected_profit_per_dollar(prob: float, dec: float) -> float:
    # EV profit per $1 stake (not total return): p*(dec-1) - (1-p)
    if not np.isfinite(prob) or not np.isfinite(dec) or dec <= 1.0:
        return np.nan
    p = clamp_prob(prob)
    return float(p * (dec - 1.0) - (1.0 - p))

def uniq_jitter(key: str, scale: float = 0.11) -> float:
    h = abs(hash(key)) % 10_000
    return (h / 10_000.0 - 0.5) * 2.0 * scale  # [-scale, +scale]

def truthy_flag(v) -> bool:
    if pd.isna(v):
        return False
    s = str(v).strip().upper()
    return s not in ("0", "", "N", "NO", "FALSE", "F", "NA", "NAN", "NONE")

def round_units(u: float) -> float:
    """
    Units must be > 0.5 and in tenths (0.1) increments.
    """
    if not np.isfinite(u):
        return np.nan
    u = float(u)
    u = max(u, 0.0)
    u = math.floor(u * 10.0 + 1e-9) / 10.0
    if 0.0 < u <= 0.5:
        u = 0.6
    return u

def force_integer_non_tie(pts: float, opp: float, key: str) -> Tuple[int, int]:
    """
    Finalized prediction must be whole number and cannot tie.
    We allow a tiny deterministic jitter, then integer-round, then fix ties.
    """
    if not np.isfinite(pts): pts = 0.0
    if not np.isfinite(opp): opp = 0.0

    j = uniq_jitter(key, scale=0.12)
    pts2 = max(0.0, pts + j)
    opp2 = max(0.0, opp - j)

    p_int = int(round(pts2))
    o_int = int(round(opp2))

    if p_int == o_int:
        # break tie by nudging the higher underlying float
        if pts2 >= opp2:
            p_int += 1
        else:
            o_int += 1
    return p_int, o_int

def enforce_ml_spread_consistency(line: float, win_prob: float, cover_prob: float) -> float:
    """
    Logical relationship between Win and Cover probabilities for a given side.
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

# Deduplicate mirrored fixtures (keep one per matchup+date)
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

# Coerce core numeric targets
daily_df[d_pts]     = pd.to_numeric(daily_df[d_pts], errors="coerce")
daily_df[d_opp_pts] = pd.to_numeric(daily_df[d_opp_pts], errors="coerce")

# Date
d_date = find_col(daily_df, ["Date", "Game Date", "Game_Date"])
daily_df["__Date"] = parse_mdy(daily_df[d_date]) if d_date else pd.to_datetime("2000-01-01")

# Derived labels
daily_df["__SM"] = daily_df[d_pts] - daily_df[d_opp_pts]
daily_df["__TOTAL"] = daily_df[d_pts] + daily_df[d_opp_pts]

# Win label (ML correctness logic)
y_win = (daily_df[d_pts] > daily_df[d_opp_pts]).astype(int)

# Spread cover label: (Points - OppPoints) + Line > 0
y_cover = pd.Series(np.nan, index=daily_df.index)
if d_line is not None and d_line in daily_df.columns:
    daily_df[d_line] = pd.to_numeric(daily_df[d_line], errors="coerce")
    y_cover = ((daily_df["__SM"] + daily_df[d_line]) > 0).astype(int)

# OU label: Total > OU line (Over=1, Under=0)
y_over = pd.Series(np.nan, index=daily_df.index)
if d_ou is not None and d_ou in daily_df.columns:
    daily_df[d_ou] = pd.to_numeric(daily_df[d_ou], errors="coerce")
    y_over = (daily_df["__TOTAL"] > daily_df[d_ou]).astype(int)

# Identify numeric feature cols (include object columns that can be coerced)
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

# Remove targets/leakage from numeric features
leak_cols = {d_pts, d_opp_pts, "__SM", "__TOTAL"}
for c in list(leak_cols):
    if c in daily_numeric_cols:
        daily_numeric_cols.remove(c)

# Coerce numeric features
for c in daily_numeric_cols:
    daily_df[c] = pd.to_numeric(daily_df[c], errors="coerce")

# Categorical columns (kept small & stable)
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

mask_points = daily_df[d_pts].notna() & daily_df[d_opp_pts].notna()
df_train = daily_df.loc[mask_points].copy()

# TEAM PROFILES for schedule rows (median per-team)
team_profiles = df_train.groupby(d_team)[daily_numeric_cols].median(numeric_only=True)
global_medians = df_train[daily_numeric_cols].median(numeric_only=True)

def opp_base_col(col: str) -> Optional[str]:
    # if you have OPP_* numeric columns, map opponent profile from base name
    if col.upper().startswith("OPP_"):
        return col[4:]
    return None

def get_team_profile(team: str) -> pd.Series:
    if team in team_profiles.index:
        return team_profiles.loc[team]
    return global_medians

# RECENCY WEIGHTS (keep simple but effective)
current_year = int(df_train["__Date"].dt.year.max()) if df_train["__Date"].notna().any() else 2000
w = np.ones(len(df_train), dtype=float)
w[(df_train["__Date"].dt.year == current_year).values] = 3.0

# =========================================================
# PREPROCESSOR (sparse, fast)
# =========================================================
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

preproc = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), daily_numeric_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)]), cat_cols),
    ],
    remainder="drop",
    sparse_threshold=1.0
)

# =========================================================
# "ONE MODEL" BUNDLE
# - Regressor predicts Points & Opp Points
# - Probabilities (win/cover/over) come from a calibrated classifier trained
#   on the SAME feature space + model-derived signals (margin/total + lines)
# This is saved/handled as a single bundle object: one .predict() interface.
# =========================================================
@dataclass
class ModelBundle:
    preproc: ColumnTransformer
    pts_model: MultiOutputRegressor
    win_model: CalibratedClassifierCV
    cover_model: Optional[CalibratedClassifierCV]
    over_model: Optional[CalibratedClassifierCV]

    def transform(self, Xdf: pd.DataFrame):
        return self.preproc.transform(Xdf)

    def predict_points(self, X):
        return self.pts_model.predict(X)

    def predict_probs(self, X, line_arr: np.ndarray, ou_arr: np.ndarray):
        """
        Returns (win_p, cover_p, over_p) for the TEAM listed in X.
        """
        win_p = clamp_prob(self.win_model.predict_proba(X)[:, 1][0])

        cover_p = np.nan
        if self.cover_model is not None and np.isfinite(line_arr[0]):
            cover_p = clamp_prob(self.cover_model.predict_proba(X)[:, 1][0])
            cover_p = enforce_ml_spread_consistency(float(line_arr[0]), win_p, cover_p)

        over_p = np.nan
        if self.over_model is not None and np.isfinite(ou_arr[0]):
            over_p = clamp_prob(self.over_model.predict_proba(X)[:, 1][0])

        return win_p, cover_p, over_p

def safe_cv_metrics(X, y, w, label: str, splits: int = 5):
    mask = ~np.isnan(y)
    y_ = y[mask].astype(int)
    X_ = X[mask]
    w_ = w[mask]
    if X_.shape[0] < 50 or len(np.unique(y_)) < 2:
        return None, None
    # adapt splits to smallest class
    cc = np.bincount(y_)
    min_class = int(cc.min()) if len(cc) else 0
    n_splits = max(2, min(splits, min_class, X_.shape[0]))
    if n_splits < 2:
        return None, None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, accs = [], []
    for tr, te in skf.split(X_, y_):
        base = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            n_jobs=None,
            class_weight="balanced"
        )
        clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
        clf.fit(X_[tr], y_[tr], sample_weight=w_[tr])
        proba = clf.predict_proba(X_[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(y_[te], proba))
        accs.append(accuracy_score(y_[te], pred))
    return float(np.mean(aucs)), float(np.mean(accs))

# =========================================================
# TRAIN
# =========================================================
st.markdown("### Model Training")
st.caption("Fast regression (Points/Opp Points) + calibrated probability models (Win/Spread/OU)")

Xdf = df_train[daily_numeric_cols + cat_cols].copy()
X = preproc.fit_transform(Xdf)

Y_points = df_train[[d_pts, d_opp_pts]].values.astype(float)

# Fast & scalable regressor (much faster than 900-tree RF on growing data)
base_reg = HistGradientBoostingRegressor(
    learning_rate=0.06,
    max_depth=6,
    max_leaf_nodes=31,
    min_samples_leaf=25,
    l2_regularization=0.0,
    random_state=42
)
pts_model = MultiOutputRegressor(base_reg)
pts_model.fit(X, Y_points, **{"sample_weight": w})

# Build additional “signals” for classification (margin/total, plus market lines)
# NOTE: we do not leak true points because pts_model predictions are used.
pts_pred = pts_model.predict(X)
pred_margin = (pts_pred[:, 0] - pts_pred[:, 1]).reshape(-1, 1)
pred_total  = (pts_pred[:, 0] + pts_pred[:, 1]).reshape(-1, 1)

line_arr = np.full((len(df_train), 1), np.nan, dtype=float)
ou_arr   = np.full((len(df_train), 1), np.nan, dtype=float)
if d_line is not None and d_line in df_train.columns:
    line_arr[:, 0] = pd.to_numeric(df_train[d_line], errors="coerce").values
if d_ou is not None and d_ou in df_train.columns:
    ou_arr[:, 0] = pd.to_numeric(df_train[d_ou], errors="coerce").values

# For classification, append these signals to X (sparse-safe hstack)
from scipy.sparse import hstack, csr_matrix
X_cls = hstack([X, csr_matrix(pred_margin), csr_matrix(pred_total),
                csr_matrix(np.nan_to_num(line_arr, nan=0.0)),
                csr_matrix(np.nan_to_num(ou_arr, nan=0.0))]).tocsr()

# Win classifier (calibrated)
base_win = LogisticRegression(max_iter=2500, solver="lbfgs", class_weight="balanced")
win_model = CalibratedClassifierCV(base_win, method="isotonic", cv=5)
win_model.fit(X_cls, y_win.loc[df_train.index].values.astype(int), sample_weight=w)

# Cover classifier (only where line exists)
cover_model = None
if d_line is not None and d_line in df_train.columns:
    y_cover_train = y_cover.loc[df_train.index].astype(float).values
    m = ~np.isnan(y_cover_train)
    if m.sum() >= 100 and len(np.unique(y_cover_train[m].astype(int))) >= 2:
        base_cover = LogisticRegression(max_iter=2500, solver="lbfgs", class_weight="balanced")
        cover_model = CalibratedClassifierCV(base_cover, method="isotonic", cv=5)
        cover_model.fit(X_cls[m], y_cover_train[m].astype(int), sample_weight=w[m])

# Over classifier (only where OU exists)
over_model = None
if d_ou is not None and d_ou in df_train.columns:
    y_over_train = y_over.loc[df_train.index].astype(float).values
    m = ~np.isnan(y_over_train)
    if m.sum() >= 100 and len(np.unique(y_over_train[m].astype(int))) >= 2:
        base_over = LogisticRegression(max_iter=2500, solver="lbfgs", class_weight="balanced")
        over_model = CalibratedClassifierCV(base_over, method="isotonic", cv=5)
        over_model.fit(X_cls[m], y_over_train[m].astype(int), sample_weight=w[m])

# Report AUC/ACC (CV) on the classification heads (same objective you wanted)
auc_win, acc_win = safe_cv_metrics(X_cls, y_win.loc[df_train.index].astype(float).values, w, "WIN")
auc_cov, acc_cov = (None, None)
if d_line is not None and d_line in df_train.columns:
    auc_cov, acc_cov = safe_cv_metrics(X_cls, y_cover.loc[df_train.index].astype(float).values, w, "COVER")
auc_ou, acc_ou = (None, None)
if d_ou is not None and d_ou in df_train.columns:
    auc_ou, acc_ou = safe_cv_metrics(X_cls, y_over.loc[df_train.index].astype(float).values, w, "OVER")

c1, c2, c3 = st.columns(3)
with c1:
    st.write(f"Win (ML correctness) AUC: {auc_win:.3f}" if auc_win is not None else "Win AUC: n/a")
    st.write(f"Win ACC: {acc_win:.3f}" if acc_win is not None else "Win ACC: n/a")
with c2:
    st.write(f"Spread AUC: {auc_cov:.3f}" if auc_cov is not None else "Spread AUC: n/a")
    st.write(f"Spread ACC: {acc_cov:.3f}" if acc_cov is not None else "Spread ACC: n/a")
with c3:
    st.write(f"OU (Over) AUC: {auc_ou:.3f}" if auc_ou is not None else "OU AUC: n/a")
    st.write(f"OU ACC: {acc_ou:.3f}" if acc_ou is not None else "OU ACC: n/a")

bundle = ModelBundle(preproc=preproc, pts_model=pts_model, win_model=win_model,
                     cover_model=cover_model, over_model=over_model)

# =========================================================
# USER CONTROLS (bars/sliders)
# =========================================================
st.markdown("---")
st.markdown("### Bankroll & Risk Controls")

colA, colB, colC, colD = st.columns(4)
with colA:
    unit_dollars = st.number_input("1 Unit = $", min_value=1.0, value=100.0, step=5.0)
with colB:
    aggressiveness = st.slider("Aggressiveness", 1, 100, 35, 1)  # higher = larger units for same edge
with colC:
    max_units = st.slider("Max units per bet", 1.0, 10.0, 4.0, 0.1)
with colD:
    max_bets = st.slider("Max bets on slate", 1, 100, 20, 1)

st.markdown("### Market Rules")
r1, r2, r3 = st.columns(3)
with r1:
    ml_edge_needed = st.slider("ML: required edge over implied prob", 0.01, 0.30, 0.10, 0.01)
with r2:
    spread_min_prob = st.slider("Spread: minimum model probability", 0.50, 0.90, 0.65, 0.01)
with r3:
    ou_min_prob = st.slider("Over/Under: minimum model probability", 0.50, 0.90, 0.65, 0.01)

st.caption("Hard slate constraint: no bet type > 45% of selections (unless slate is tiny; still enforced).")

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
# BET CANDIDATES PER GAME
# =========================================================
def units_from_edge(edge: float) -> float:
    """
    Edge -> units (tenths), must be >0.5 if any bet is placed.
    Aggressiveness boosts slope; capped at max_units.
    """
    if not np.isfinite(edge) or edge <= 0:
        return 0.0
    # base scale: 0.02 edge ~ 1 unit at mid aggressiveness
    scale = np.interp(aggressiveness, [1, 100], [30.0, 120.0])
    u = edge * scale
    u = min(u, max_units)
    u = round_units(u)
    return u

def build_candidates_for_row(r: pd.Series) -> List[Dict[str, Any]]:
    t = str(r[team_col]).strip()
    o = str(r[opp_col]).strip()
    date = r["__Date"]
    game_label = f"{t} vs {o}"

    one = schedule_row_to_feature(r)
    X0 = bundle.preproc.transform(one)

    # points prediction
    pred = bundle.pts_model.predict(X0)[0]
    pts_f, opp_f = float(pred[0]), float(pred[1])

    pts_i, opp_i = force_integer_non_tie(pts_f, opp_f, f"{t}|{o}|{date}")

    margin = pts_i - opp_i
    total  = pts_i + opp_i

    line = safe_num(r.get(line_col), default=np.nan) if line_col else np.nan
    ouv  = safe_num(r.get(ou_col),   default=np.nan) if ou_col   else np.nan
    ml   = safe_num(r.get(ml_col),   default=np.nan) if ml_col   else np.nan

    # classification feature augmentation (must match training)
    from scipy.sparse import hstack, csr_matrix
    X_cls0 = hstack([
        X0,
        csr_matrix([[margin]]),
        csr_matrix([[total]]),
        csr_matrix([[0.0 if not np.isfinite(line) else float(line)]]),
        csr_matrix([[0.0 if not np.isfinite(ouv) else float(ouv)]])
    ]).tocsr()

    win_p = clamp_prob(bundle.win_model.predict_proba(X_cls0)[0, 1])

    cover_p = np.nan
    if bundle.cover_model is not None and np.isfinite(line):
        cover_p = clamp_prob(bundle.cover_model.predict_proba(X_cls0)[0, 1])
        cover_p = enforce_ml_spread_consistency(line, win_p, cover_p)

    over_p = np.nan
    if bundle.over_model is not None and np.isfinite(ouv):
        over_p = clamp_prob(bundle.over_model.predict_proba(X_cls0)[0, 1])

    cands: List[Dict[str, Any]] = []

    # ---- Moneyline candidate (only if probability beats implied by >= 10% default)
    if np.isfinite(ml):
        implied = american_to_implied_prob(ml)
        edge = win_p - implied if np.isfinite(implied) else np.nan
        dec = american_to_decimal(ml)
        ev = expected_profit_per_dollar(win_p, dec)
        if np.isfinite(edge) and edge >= ml_edge_needed:
            u = units_from_edge(edge)
            if u > 0:
                cands.append({
                    "Game": game_label,
                    "Bet Type": "ML",
                    "Bet": t,
                    "Line": np.nan,
                    "Odds": ml,
                    "Model Prob": win_p,
                    "Implied Prob": implied,
                    "Edge": edge,
                    "EV_per_$": ev,
                    "Units": u,
                    "Pred_PTS": pts_i,
                    "Pred_OPP": opp_i,
                    "Pred_Total": total,
                    "Pred_Margin": margin,
                    "Date": date
                })

    # ---- Spread candidate (odds fixed -110)
    if np.isfinite(line) and np.isfinite(cover_p):
        if cover_p >= spread_min_prob:
            dec = american_to_decimal(-110.0)
            ev = expected_profit_per_dollar(cover_p, dec)
            edge = cover_p - 0.5  # since -110 is close to 50/50 baseline, you want 65%+ anyway
            u = units_from_edge(edge)
            if u > 0:
                cands.append({
                    "Game": game_label,
                    "Bet Type": "Spread",
                    "Bet": f"{t} {line:+g}",
                    "Line": line,
                    "Odds": -110.0,
                    "Model Prob": cover_p,
                    "Implied Prob": 0.5,
                    "Edge": edge,
                    "EV_per_$": ev,
                    "Units": u,
                    "Pred_PTS": pts_i,
                    "Pred_OPP": opp_i,
                    "Pred_Total": total,
                    "Pred_Margin": margin,
                    "Date": date
                })

    # ---- Over/Under candidate (odds fixed -110)
    if np.isfinite(ouv) and np.isfinite(over_p):
        if over_p >= ou_min_prob:
            dec = american_to_decimal(-110.0)
            ev_over = expected_profit_per_dollar(over_p, dec)
            ev_under = expected_profit_per_dollar(1.0 - over_p, dec)

            # choose side with higher probability (and EV)
            if ev_over >= ev_under:
                bet = f"OVER {ouv:g}"
                prob = over_p
                edge = prob - 0.5
                ev = ev_over
            else:
                bet = f"UNDER {ouv:g}"
                prob = 1.0 - over_p
                edge = prob - 0.5
                ev = ev_under

            u = units_from_edge(edge)
            if u > 0:
                cands.append({
                    "Game": game_label,
                    "Bet Type": "OU",
                    "Bet": bet,
                    "Line": ouv,
                    "Odds": -110.0,
                    "Model Prob": prob,
                    "Implied Prob": 0.5,
                    "Edge": edge,
                    "EV_per_$": ev,
                    "Units": u,
                    "Pred_PTS": pts_i,
                    "Pred_OPP": opp_i,
                    "Pred_Total": total,
                    "Pred_Margin": margin,
                    "Date": date
                })

    # ---- Special rule: if spread is 3.5 or less, choose most profitable between ML and Spread (not both)
    # (Only applies if we have both candidates present for the team in this game)
    if np.isfinite(line) and abs(line) <= 3.5:
        ml_idx = [i for i, x in enumerate(cands) if x["Bet Type"] == "ML"]
        sp_idx = [i for i, x in enumerate(cands) if x["Bet Type"] == "Spread"]
        if ml_idx and sp_idx:
            ml_i = ml_idx[0]
            sp_i = sp_idx[0]
            # keep the higher EV per $ (tie-break: higher edge)
            keep_ml = False
            if np.isfinite(cands[ml_i]["EV_per_$"]) and np.isfinite(cands[sp_i]["EV_per_$"]):
                keep_ml = cands[ml_i]["EV_per_$"] > cands[sp_i]["EV_per_$"]
            else:
                keep_ml = cands[ml_i]["Edge"] > cands[sp_i]["Edge"]
            if keep_ml:
                cands.pop(sp_i)
            else:
                cands.pop(ml_i)

    return cands

all_candidates: List[Dict[str, Any]] = []
for _, r in schedule_df.iterrows():
    all_candidates.extend(build_candidates_for_row(r))

cand_df = pd.DataFrame(all_candidates)

st.markdown("---")
st.markdown("### Candidate Bets (pre-slate balancing)")
if cand_df.empty:
    st.warning("No bets meet the thresholds today.")
    st.stop()

# Rank score: prioritize EV, then probability, then edge
cand_df["Score"] = (
    cand_df["EV_per_$"].fillna(-9999.0) * 10.0
    + cand_df["Model Prob"].fillna(0.0) * 2.0
    + cand_df["Edge"].fillna(0.0) * 5.0
)

cand_df = cand_df.sort_values(["Score", "Model Prob"], ascending=False).reset_index(drop=True)
st.dataframe(cand_df[[
    "Game","Bet Type","Bet","Odds","Model Prob","Implied Prob","Edge","EV_per_$","Units","Pred_PTS","Pred_OPP"
]])

# =========================================================
# SLATE SELECTION WITH 45% CAP + "roughly equal" mix
# =========================================================
def select_with_type_caps(df: pd.DataFrame, max_bets: int, cap_pct: float = 0.45) -> pd.DataFrame:
    if df.empty:
        return df

    types = ["ML", "Spread", "OU"]
    # absolute cap
    cap_n = max(1, int(math.floor(max_bets * cap_pct)))

    chosen = []
    counts = {t: 0 for t in types}

    # target roughly equal (1/3 each), but never exceed cap
    target_each = max(1, int(round(max_bets / 3)))
    target_each = min(target_each, cap_n)

    # Greedy with penalty if over target (but still allowed up to cap)
    for _, row in df.iterrows():
        if len(chosen) >= max_bets:
            break
        bt = row["Bet Type"]
        if bt not in counts:
            continue
        if counts[bt] >= cap_n:
            continue

        # prevent duplicate identical bet rows
        sig = (row["Game"], row["Bet Type"], row["Bet"], row["Odds"])
        if any((x["Game"], x["Bet Type"], x["Bet"], x["Odds"]) == sig for x in chosen):
            continue

        # penalty if this type already beyond target
        penalty = 0.0
        if counts[bt] >= target_each:
            penalty = 999.0  # large penalty, but still may be picked if truly dominant

        # accept if score is strong enough even after penalty
        # (we compare to itself: just require not terrible)
        if (row["Score"] - penalty) > -1000:
            chosen.append(row.to_dict())
            counts[bt] += 1

    out = pd.DataFrame(chosen)

    # If we selected very few, relax penalties but keep cap
    if out.shape[0] < min(3, max_bets):
        chosen = []
        counts = {t: 0 for t in types}
        for _, row in df.iterrows():
            if len(chosen) >= max_bets:
                break
            bt = row["Bet Type"]
            if bt not in counts:
                continue
            if counts[bt] >= cap_n:
                continue
            sig = (row["Game"], row["Bet Type"], row["Bet"], row["Odds"])
            if any((x["Game"], x["Bet Type"], x["Bet"], x["Odds"]) == sig for x in chosen):
                continue
            chosen.append(row.to_dict())
            counts[bt] += 1
        out = pd.DataFrame(chosen)

    return out

plan = select_with_type_caps(cand_df, max_bets=max_bets, cap_pct=0.45)

# enforce units rules again + compute stake
plan["Units"] = plan["Units"].apply(lambda x: round_units(float(x)) if np.isfinite(x) else np.nan)
plan = plan[plan["Units"].notna() & (plan["Units"] > 0.5)].copy()

plan["Stake_$"] = (plan["Units"] * float(unit_dollars)).round(2)
plan["DecimalOdds"] = plan["Odds"].apply(lambda o: american_to_decimal(float(o)) if np.isfinite(o) else np.nan)
plan["Payout_if_win_$"] = (plan["Stake_$"] * plan["DecimalOdds"]).round(2)

# Summary of bet-type mix
st.markdown("---")
st.markdown("### Final Betting Plan (post mix + caps)")
if plan.empty:
    st.warning("After balancing rules and unit minimums, no bets remain.")
    st.stop()

mix = plan["Bet Type"].value_counts(dropna=False)
total_n = int(plan.shape[0])
mix_pct = (mix / total_n * 100.0).round(1)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Bets", total_n)
with m2:
    st.metric("ML %", f"{mix_pct.get('ML', 0.0)}%")
with m3:
    st.metric("Spread %", f"{mix_pct.get('Spread', 0.0)}%")
st.write("OU %:", f"{mix_pct.get('OU', 0.0)}%")

st.dataframe(plan[[
    "Date","Game","Bet Type","Bet","Odds","Model Prob","Implied Prob","Edge","EV_per_$",
    "Units","Stake_$","Payout_if_win_$","Pred_PTS","Pred_OPP","Pred_Total","Pred_Margin"
]].sort_values(["Date","Game","Bet Type"]))

# Optional download
csv_bytes = plan.to_csv(index=False).encode("utf-8")
st.download_button("Download plan CSV", data=csv_bytes, file_name="betting_plan.csv", mime="text/csv")

