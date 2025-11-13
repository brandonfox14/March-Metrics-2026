# APP/Pages/9_Betting.py
# =========================================================
# Betting Page — schedule scoring uses team/opponent profiles
# and recency-weighted ML models for Win / Spread / Total
# =========================================================
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
from sklearn.metrics import average_precision_score, roc_auc_score

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

def decimal_to_american(dec: float) -> str:
    if not np.isfinite(dec) or dec <= 1.0:
        return "N/A"
    if dec >= 2.0:
        return f"+{int(round((dec - 1.0) * 100))}"
    return f"{int(round(-100.0 / (dec - 1.0)))}"

def kelly_fraction(prob: float, dec_odds: float, cap: float = 1.0) -> float:
    """Uncapped Kelly (we'll normalize across slate to diversify)."""
    if not np.isfinite(prob) or not np.isfinite(dec_odds) or dec_odds <= 1.0:
        return 0.0
    p = max(0.0, min(1.0, prob))
    b = dec_odds - 1.0
    if b <= 0:
        return 0.0
    f = (p * (b + 1.0) - 1.0) / b
    if f <= 0:
        return 0.0
    return float(min(cap, f))

def safe_num(x, default=0.0) -> float:
    try:
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default

def clean_rank_value(x) -> float:
    try:
        v = float(x)
        if np.isnan(v): return 51.0
        return v
    except Exception:
        return 51.0

def uniq_jitter(key: str, scale: float = 0.11) -> float:
    """Slight deterministic jitter so scores aren't identical."""
    h = abs(hash(key)) % 10_000
    return (h / 10_000.0 - 0.5) * 2.0 * scale  # [-scale, +scale]

def zero_as_missing_mask(df: pd.DataFrame, numeric_cols: List[str], frac_threshold: float = 0.6) -> List[str]:
    cols = []
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            zfrac = (s == 0).mean()
            if zfrac > frac_threshold:
                cols.append(c)
    return cols

def impute_numeric_with_zero_missing(df: pd.DataFrame, numeric_cols: List[str], zero_missing_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in zero_missing_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        out.loc[out[c] == 0, c] = np.nan
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

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

# =========================================================
# LOAD DATA
# =========================================================
schedule_df_raw = load_csv(SCHEDULE_FILE)  # raw for drilldown
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

# Deduplicate mirrored fixtures for scoring (keep one), but retain raw for drilldown
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

# zero→NaN heuristic and numeric coercion (for team profiles)
zero_missing_cols = zero_as_missing_mask(daily_df, daily_numeric_cols, frac_threshold=0.6)
daily_df_num = impute_numeric_with_zero_missing(daily_df, daily_numeric_cols, zero_missing_cols)

# TEAM PROFILES (median per-team across all numeric features)
team_profiles = daily_df_num.groupby(d_team)[daily_numeric_cols].median(numeric_only=True)
global_medians = daily_df_num[daily_numeric_cols].median(numeric_only=True)

def get_team_profile(team: str) -> pd.Series:
    if team in team_profiles.index:
        return team_profiles.loc[team]
    return global_medians

def opp_base_col(col: str) -> Optional[str]:
    if col.upper().startswith("OPP_"):
        return col[4:]
    return None

# --- Targets & Recency for training ---
# Date
d_date = find_col(daily_df, ["Date", "Game Date", "Game_Date"])
if d_date is None:
    daily_df["__Date"] = pd.to_datetime("2000-01-01")
else:
    daily_df["__Date"] = parse_mdy(daily_df[d_date])

# Points
daily_df[d_pts]     = pd.to_numeric(daily_df[d_pts], errors="coerce")
daily_df[d_opp_pts] = pd.to_numeric(daily_df[d_opp_pts], errors="coerce")

# SM and TOTAL
sm_col = find_col(daily_df, ["SM", "Spread Margin"])
if sm_col is None:
    daily_df["__SM"] = daily_df[d_pts] - daily_df[d_opp_pts]
    sm_col = "__SM"

tot_col = find_col(daily_df, ["Total Points", "TOTAL_POINTS", "Total"])
if tot_col is None:
    daily_df["__TOTAL"] = daily_df[d_pts] + daily_df[d_opp_pts]
    tot_col = "__TOTAL"

# Moneyline win label
y_win = (daily_df[d_pts] > daily_df[d_opp_pts]).astype(int)

# Spread cover label: SM + Line > 0
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

# Only rows with point targets for regression
mask_points = daily_df[d_pts].notna() & daily_df[d_opp_pts].notna()
df_train = daily_df.loc[mask_points].copy()

# RECENCY WEIGHTS (current season heavier)
current_year = int(df_train["__Date"].dt.year.max())
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
Y_points = df_train[[d_pts, d_opp_pts]].values

# Align labels with df_train
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
# TRAIN MODELS — Regression + 3 Classifiers
# =========================================================
st.markdown("### Model Training")
st.caption("Regression for Points/Opp Points + classifiers for Win, Spread, and Total with recency-weighted training.")

# Regression: points and opp points
rf_points = RandomForestRegressor(
    n_estimators=900,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=1,
    max_features="sqrt"
)
rf_points.fit(X_all, Y_points, sample_weight=sample_weight)

def cv_auc(X: np.ndarray, y: np.ndarray, w: np.ndarray, splits: int = 5) -> Optional[float]:
    mask = ~np.isnan(y)
    y_ = y[mask]
    X_ = X[mask]
    w_ = w[mask]
    if len(np.unique(y_)) < 2 or X_.shape[0] < splits:
        return None
    skf = StratifiedKFold(n_splits=min(splits, X_.shape[0]))
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

# AUCs
auc_win   = cv_auc(X_all, y_win_train.astype(float), sample_weight)
auc_cover = cv_auc(X_all, y_cover_train, sample_weight)
auc_over  = cv_auc(X_all, y_over_train, sample_weight)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.write(f"Moneyline / Win AUC: {auc_win:.3f}" if auc_win is not None else "Moneyline / Win AUC: n/a")
with col_m2:
    st.write(f"Spread AUC: {auc_cover:.3f}" if auc_cover is not None else "Spread AUC: n/a")
with col_m3:
    st.write(f"Total (Over/Under) AUC: {auc_over:.3f}" if auc_over is not None else "Total (Over/Under) AUC: n/a")

# Final classifiers fit on all training data
clf_win = RandomForestClassifier(
    n_estimators=800, random_state=42, class_weight="balanced_subsample", n_jobs=-1
)
clf_win.fit(X_all, y_win_train, sample_weight=sample_weight)

clf_cover = RandomForestClassifier(
    n_estimators=700, random_state=42, class_weight="balanced_subsample", n_jobs=-1
)
mask_cover_fit = ~np.isnan(y_cover_train)
if mask_cover_fit.any():
    clf_cover.fit(X_all[mask_cover_fit], y_cover_train[mask_cover_fit].astype(int),
                  sample_weight=sample_weight[mask_cover_fit])
else:
    clf_cover = None

clf_over = RandomForestClassifier(
    n_estimators=700, random_state=42, class_weight="balanced_subsample", n_jobs=-1
)
mask_over_fit = ~np.isnan(y_over_train)
if mask_over_fit.any():
    clf_over.fit(X_all[mask_over_fit], y_over_train[mask_over_fit].astype(int),
                 sample_weight=sample_weight[mask_over_fit])
else:
    clf_over = None

# Quick PR-AUC for win (optional, sanity check)
def cv_ap_class(X, y, splits=5) -> Optional[float]:
    if len(np.unique(y)) < 2 or X.shape[0] < splits:
        return None
    skf = StratifiedKFold(n_splits=min(splits, X.shape[0]))
    aps = []
    for tr, te in skf.split(X, y):
        m = RandomForestClassifier(
            n_estimators=500, random_state=42, class_weight="balanced_subsample", n_jobs=-1
        )
        m.fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])[:, 1]
        aps.append(average_precision_score(y[te], proba))
    return float(np.mean(aps)) if aps else None

st.caption("Approx PR-AUC (Win):")
ap_win = cv_ap_class(X_all, y_win_train)
st.write(f"Win (PR-AUC ~): {ap_win:.3f}" if ap_win is not None else "Win (PR-AUC): n/a")

# =========================================================
# USER CONTROLS
# =========================================================
st.markdown("---")
st.markdown("### Bankroll & Strategy")
colA, colB, colC = st.columns(3)
with colA:
    units_base = st.number_input("Units of betting", min_value=1.0, value=100.0, step=1.0)
    bad_day_target = st.number_input("Units floor (bad day, 95% target)", min_value=0.0, value=65.5, step=0.5)
with colB:
    consistent_day_target = st.number_input("Units target (70% day)", min_value=0.0, value=140.0, step=1.0)
    homerun = st.slider("Homerun hitter (parlay aggressiveness)", 1, 10, 7)
with colC:
    ladder_start = st.selectbox("Ladder Starter", ["No","Yes"], index=1)
    ladder_cont  = st.selectbox("Ladder Continuation", ["No","Yes"], index=0)

# Ladder amounts (starter ≤ 10% bankroll, continuation free entry)
max_ladder_start = max(0.0, round(units_base * 0.10, 2))
colL1, colL2 = st.columns(2)
with colL1:
    ladder_start_amt = st.slider("Ladder Starter Units (≤10% bankroll)", 0.0, float(max_ladder_start), min(float(max_ladder_start), 5.0), 0.5)
with colL2:
    ladder_cont_amt = st.number_input("Ladder Continuation Units (excluded from slate)", min_value=0.0, value=0.0, step=0.5)

colD, colE = st.columns(2)
with colD:
    min_bets, max_bets = st.slider("Bets minimum / maximum", 4, 100, (10, 25))
with colE:
    st.caption("Min stake per bet is 0.5u. No hard per-bet cap; diversification via edges.")

# Effective slate bankroll excludes both ladder buckets
effective_units = max(0.0, units_base - (ladder_start_amt if ladder_start == "Yes" else 0.0) - (ladder_cont_amt if ladder_cont == "Yes" else 0.0))
st.info(f"Effective units for slate (excludes ladders): {effective_units:.2f}")

# =========================================================
# SCHEDULE → FEATURE with TEAM/OPP PROFILES
# =========================================================
def schedule_row_to_feature(row: pd.Series) -> pd.DataFrame:
    t = str(row[team_col]).strip()
    o = str(row[opp_col]).strip()
    team_prof = get_team_profile(t)
    opp_prof  = get_team_profile(o)
    data: Dict[str, Any] = {}
    # numeric fields
    for c in daily_numeric_cols:
        val = row.get(c) if c in row.index else np.nan
        val = pd.to_numeric(val, errors="coerce")
        if pd.notna(val):
            data[c] = float(val)
            continue
        base = opp_base_col(c)
        if base is not None:
            data[c] = float(opp_prof.get(base, global_medians.get(c, np.nan)))
            continue
        data[c] = float(team_prof.get(c, global_medians.get(c, np.nan)))
    # categorical fields
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
# SCORE GAMES
# =========================================================
rows = []
for _, r in schedule_df.iterrows():
    t = str(r[team_col]).strip()
    o = str(r[opp_col]).strip()

    one = schedule_row_to_feature(r)
    X = preproc.transform(one)

    # points prediction
    pts_pred = rf_points.predict(X)[0]
    pts, oppp = float(pts_pred[0]), float(pts_pred[1])

    # jitter to avoid identical predictions
    jitter = uniq_jitter(f"{t}|{o}|{r['__Date']}")
    pts  = max(0.0, pts  + jitter)
    oppp = max(0.0, oppp - jitter)

    margin = pts - oppp
    total  = pts + oppp

    # probabilities from classifiers
    win_p = clf_win.predict_proba(X)[0, 1]
    win_p = min(0.995, max(0.005, float(win_p)))

    line = safe_num(r.get(line_col), default=np.nan) if line_col else np.nan
    ouv  = safe_num(r.get(ou_col),   default=np.nan) if ou_col   else np.nan
    mline= safe_num(r.get(ml_col),   default=np.nan) if ml_col   else np.nan

    cover_prob = np.nan
    if np.isfinite(line) and clf_cover is not None:
        cover_prob = clf_cover.predict_proba(X)[0,1]
        cover_prob = min(0.995, max(0.005, float(cover_prob)))

    over_prob = np.nan
    if np.isfinite(ouv) and clf_over is not None:
        over_prob = clf_over.predict_proba(X)[0,1]
        over_prob = min(0.995, max(0.005, float(over_prob)))

    # Moneyline edge & Kelly
    ml_dec = american_to_decimal(mline) if np.isfinite(mline) else np.nan
    ml_edge = np.nan
    ml_kelly = 0.0
    if np.isfinite(ml_dec):
        ml_edge  = win_p - (1.0 / ml_dec)
        ml_kelly = kelly_fraction(win_p, ml_dec, cap=1.0)

    # Spread edge & Kelly (default -110, decimal ~1.909)
    spread_dec = 1.909 if np.isfinite(line) else np.nan
    spread_edge = (cover_prob - (1.0 / spread_dec)) if (np.isfinite(spread_dec) and np.isfinite(cover_prob)) else np.nan
    spread_kelly = kelly_fraction(cover_prob, spread_dec, cap=1.0) if np.isfinite(spread_edge) else 0.0

    # OU edge & Kelly
    ou_dec = 1.909 if np.isfinite(ouv) else np.nan
    ou_pick, ou_prob, ou_edge_v, ou_kelly = None, np.nan, np.nan, 0.0
    if np.isfinite(ou_dec) and np.isfinite(over_prob):
        over_edge  = over_prob - (1.0 / ou_dec)
        under_prob = 1.0 - over_prob
        under_edge = under_prob - (1.0 / ou_dec)
        if over_edge >= under_edge:
            ou_pick, ou_prob, ou_edge_v = "Over", over_prob, over_edge
        else:
            ou_pick, ou_prob, ou_edge_v = "Under", under_prob, under_edge
        ou_kelly = kelly_fraction(ou_prob, ou_dec, cap=1.0)

    # priority flags
    is_top25 = False
    if top25_col and top25_col in schedule_df.columns:
        v = r.get(top25_col); is_top25 = (str(v).strip() not in ("0","","N","FALSE","NaN"))
    is_mm = False
    if mm_col and mm_col in schedule_df.columns:
        v = r.get(mm_col); is_mm = (str(v).strip() not in ("0","","N","FALSE","NaN"))
    conf = r.get(opp_conf_col) if (opp_conf_col and opp_conf_col in r.index) else r.get(conf_col)
    cc = str(conf).strip().upper() if conf is not None else ""
    is_p5 = cc in ("SEC","BIG TEN","B1G","BIG 12","ACC","PAC-12","PAC 12","BIG EAST")

    # choose best bet by edge, subject to rule 2 (no spread when +ML < +150)
    edges: List[Tuple[str, float, float, float, float, float]] = []

    allow_spread = True
    if np.isfinite(mline) and mline > 0 and mline < 150:
        allow_spread = False

    if np.isfinite(spread_edge) and allow_spread:
        edges.append(("Spread", spread_edge, cover_prob, spread_dec, spread_kelly, line))
    if np.isfinite(ml_edge):
        edges.append(("Moneyline", ml_edge, win_p, ml_dec, ml_kelly, mline))
    if np.isfinite(ou_edge_v):
        edges.append((ou_pick or "OU", ou_edge_v, ou_prob, ou_dec, ou_kelly, ouv))

    best = max(edges, key=lambda z: (z[1] if np.isfinite(z[1]) else -1e9), default=None)

    # mark big dog for pair: rule 3 (big +odds with good edge)
    is_big_dog_pair = False
    if np.isfinite(mline) and mline >= 150 and np.isfinite(ml_edge) and ml_edge > 0.04:
        min_dec_for_pair = float(np.interp(homerun, [1, 10], [2.5, 4.0]))  # ~+150 .. +300+
        if np.isfinite(ml_dec) and ml_dec >= min_dec_for_pair:
            is_big_dog_pair = True

    rows.append({
        "Date": r["__Date"],
        "Team": t,
        "Opponent": o,
        "Pred_Points": round(pts, 1),
        "Pred_Opp_Points": round(oppp, 1),
        "Pred_Margin": round(margin, 1),
        "Pred_Total": round(total, 1),
        "WinProb": win_p,
        "Line": line if np.isfinite(line) else None,
        "CoverProb": cover_prob if np.isfinite(cover_prob) else None,
        "ML": mline if np.isfinite(mline) else None,
        "MLProb": win_p,
        "OU_Line": ouv if np.isfinite(ouv) else None,
        "OverProb": over_prob if np.isfinite(over_prob) else None,
        "BestMarket": best[0] if best else None,
        "BestEdge": best[1] if best else np.nan,
        "BestProb": best[2] if best else np.nan,
        "BestDecOdds": best[3] if best else np.nan,
        "BestKelly": best[4] if best else 0.0,
        "BestLineVal": best[5] if best else None,
        "IsTop25": is_top25,
        "IsMM": is_mm,
        "IsP5": is_p5,
        "RowObj": r,
        "IsDogPair": is_big_dog_pair,
    })

pred_df = pd.DataFrame(rows)

# Ban Penn State from betting predictions (as the Team)
pred_df = pred_df[~pred_df["Team"].apply(is_penn_state)].reset_index(drop=True)

if pred_df.empty:
    st.info("No priced games available (after filters).")
    st.stop()

# =========================================================
# SLATE CONSTRUCTION (diversified by market type)
# =========================================================
st.markdown("---")
st.markdown("### Suggested Slate (Model-Driven)")

# Priority sort (Top25, MM, P5, Date, then edge)
pred_df["_prio"] = pred_df.apply(
    lambda r: (0 if r["IsTop25"] else 1, 0 if r["IsMM"] else 1, 0 if r["IsP5"] else 1, r["Date"]),
    axis=1
)
base = pred_df.dropna(subset=["BestEdge"]).copy()
base = base[base["BestEdge"] > 0].copy()
if base.empty:
    st.info("No +EV bets under current markets.")
    st.stop()

base = base.sort_values(by=["_prio","BestEdge"], ascending=[True, False]).reset_index(drop=True)

# Market buckets
spread_cands = base[base["BestMarket"] == "Spread"].copy()
ou_cands     = base[base["BestMarket"].isin(["Over","Under","OU"])].copy()
ml_cands     = base[base["BestMarket"] == "Moneyline"].copy()

# For ML, bias toward strong favorites (- odds) but still allow + odds with edge
if not ml_cands.empty:
    ml_cands = ml_cands.sort_values(
        by=["WinProb","BestEdge"],
        ascending=[False, False]
    )

dog_cands = base[base["IsDogPair"]].copy()  # games eligible for ML+spread pair

# Determine slate size
max_available = len(base)
N_total = min(max_bets, max(min_bets, max_available))

# Target counts by market (approximate)
spread_target = int(math.floor(0.35 * N_total))
ou_target     = int(math.floor(0.35 * N_total))
ml_target     = int(math.floor(0.20 * N_total))

base_sum = spread_target + ou_target + ml_target
remaining = N_total - base_sum
# Distribute remainder across buckets in order of richness
bucket_order = [
    ("spread", len(spread_cands)),
    ("ou", len(ou_cands)),
    ("ml", len(ml_cands)),
]
while remaining > 0:
    changed = False
    for name, size in bucket_order:
        if remaining <= 0:
            break
        if name == "spread" and spread_target < size:
            spread_target += 1; remaining -= 1; changed = True
        elif name == "ou" and ou_target < size:
            ou_target += 1; remaining -= 1; changed = True
        elif name == "ml" and ml_target < size:
            ml_target += 1; remaining -= 1; changed = True
    if not changed:
        break

# Pick from each bucket (no restriction on re-using same game; each row is unique)
slate_indices = []

def pick_from_bucket(df: pd.DataFrame, target: int, already: List[int]) -> List[int]:
    if target <= 0 or df.empty:
        return []
    df = df[~df.index.isin(already)]
    if df.empty:
        return []
    return df.head(target).index.tolist()

slate_indices += pick_from_bucket(spread_cands, spread_target, slate_indices)
slate_indices += pick_from_bucket(ou_cands,     ou_target,     slate_indices)
slate_indices += pick_from_bucket(ml_cands,     ml_target,     slate_indices)

# If we still have room, fill with best remaining regardless of market
if len(slate_indices) < N_total:
    need = N_total - len(slate_indices)
    remaining_df = base[~base.index.isin(slate_indices)]
    extra_idx = remaining_df.head(need).index.tolist()
    slate_indices += extra_idx

slate = base.loc[slate_indices].copy()
slate = slate.sort_values(by=["_prio","BestEdge"], ascending=[True, False]).reset_index(drop=True)

# Kelly-based stake sizing (normalized to effective_units, min 0.5u)
edges_pos = slate["BestEdge"].clip(lower=1e-6)
total_edge = float(edges_pos.sum()) if len(edges_pos) > 0 else 1.0
slate["Stake_raw"] = edges_pos / total_edge * effective_units
slate["Stake"] = (slate["Stake_raw"] * 2).round() / 2.0  # nearest 0.5

def renorm_half_units(df, target):
    diff = round(target - float(df["Stake"].sum()), 2)
    step = 0.5 if diff > 0 else -0.5
    idx = 0
    order = df.sort_values("BestEdge", ascending=False).index.tolist()
    while abs(diff) >= 0.49 and idx < len(order) * 5:
        i = order[idx % len(order)]
        newv = df.at[i, "Stake"] + step
        if newv >= 0.5:
            df.at[i, "Stake"] = newv
            diff = round(target - float(df["Stake"].sum()), 2)
        idx += 1
    return df

if not slate.empty:
    slate.loc[slate["Stake"] < 0.5, "Stake"] = 0.5
    slate = renorm_half_units(slate, effective_units)

def pick_label(r):
    if r["BestMarket"] in ("Over","Under","OU"):
        side = r["BestMarket"] if r["BestMarket"] in ("Over","Under") else (
            "Over" if (r["OverProb"] and r["OverProb"]>=0.5) else "Under"
        )
        val = int(r["BestLineVal"]) if r["BestLineVal"] is not None else ""
        return f"{side} {val}".strip()
    elif r["BestMarket"] == "Spread":
        use_line = r["Line"]
        if use_line is None:
            return f"{r['Team']} Spread"
        sign = "+" if use_line > 0 else ""
        return f"{r['Team']} {sign}{use_line}"
    elif r["BestMarket"] == "Moneyline":
        return f"{r['Team']} ML"
    return r["BestMarket"] or "Bet"

if slate.empty:
    st.info("No +EV bets under current markets.")
else:
    spent_units = float(slate["Stake"].sum())
    remainder = round(max(0.0, effective_units - spent_units), 2)

    plan_rows = []
    for _, r in slate.iterrows():
        game_label = f"{r['Team']} vs {r['Opponent']}"
        stake = float(r["Stake"])

        # Rule 3 UPDATED: big dog pairing (ML + Spread with 3x spread vs ML)
        if r.get("IsDogPair", False):
            # Maintain total stake while making spread = 3× ML
            ml_units = stake / 4.0
            spread_units = stake - ml_units
            ml_units = round(ml_units * 2) / 2.0
            spread_units = round(spread_units * 2) / 2.0

            if ml_units >= 0.5:
                plan_rows.append({
                    "GAME": game_label,
                    "BET TYPE": f"{r['Team']} ML (dog value)",
                    "BET AMOUNT": f"{ml_units:.1f} UNITS"
                })

            spread_line = r["Line"]
            if spread_units >= 0.5 and spread_line is not None:
                sign = "+" if spread_line > 0 else ""
                spread_label = f"{r['Team']} {sign}{spread_line}"
                plan_rows.append({
                    "GAME": game_label,
                    "BET TYPE": f"{spread_label} (spread ~3× ML stake)",
                    "BET AMOUNT": f"{spread_units:.1f} UNITS"
                })
        else:
            plan_rows.append({
                "GAME": game_label,
                "BET TYPE": pick_label(r),
                "BET AMOUNT": f"{stake:.1f} UNITS"
            })

    # Optionally allocate remainder to most probable ML favorite
    if remainder >= 0.5:
        ml_cand = pred_df[(pred_df["BestMarket"]=="Moneyline") & pred_df["BestProb"].notna()].copy()
        ml_cand = ml_cand.sort_values(["WinProb","BestEdge"], ascending=[False, False])
        if not ml_cand.empty:
            top = ml_cand.iloc[0]
            plan_rows.append({
                "GAME": f"{top['Team']} vs {top['Opponent']}",
                "BET TYPE": "Most probable ML (remainder)",
                "BET AMOUNT": f"{(round(remainder*2)/2):.1f} UNITS"
            })
            st.info("Remainder assigned to the most probable ML — end of smart bets.")

    plan_df = pd.DataFrame(plan_rows, columns=["GAME","BET TYPE","BET AMOUNT"])
    st.dataframe(plan_df, use_container_width=True)

# =========================================================
# LADDERS
# =========================================================
st.markdown("---")
st.markdown("### Ladders")

def fmt_price(dec):
    if not np.isfinite(dec):
        return "-110"
    return decimal_to_american(dec)

def pick_ladder(df: pd.DataFrame, odds_min: float, odds_max: float, exclude_key=None):
    """
    Pick best within decimal-odds window [odds_min, odds_max].
    """
    cands = []
    for _, r in df.iterrows():
        mkt = r["BestMarket"]
        if not mkt:
            continue
        key = (r["Team"], r["Opponent"], r["Date"], mkt)
        if exclude_key and key == exclude_key:
            continue
        dec = r["BestDecOdds"]
        if not np.isfinite(dec):
            if mkt in ("Spread","Over","Under","OU"):
                dec = 1.909  # -110 default
            else:
                continue
        if odds_min <= dec <= odds_max:
            if np.isfinite(r["BestProb"]):
                cands.append((r["BestProb"], r))
    if not cands:
        return None
    cands.sort(key=lambda z: z[0], reverse=True)
    return cands[0][1]

c1, c2 = st.columns(2)
ladder_start_pick, ladder_cont_pick = None, None
ex_key = None
if ladder_start == "Yes" and ladder_start_amt >= 0.5:
    ladder_start_pick = pick_ladder(pred_df, 1.83, 3.00)  # -120 .. +200
    if ladder_start_pick is not None:
        ex_key = (ladder_start_pick["Team"], ladder_start_pick["Opponent"],
                  ladder_start_pick["Date"], ladder_start_pick["BestMarket"])

if ladder_cont == "Yes" and ladder_cont_amt >= 0.5:
    ladder_cont_pick = pick_ladder(pred_df, 1.50, 2.05, exclude_key=ex_key)  # -200 .. +105

with c1:
    st.write("Ladder Starter " + ("(enabled)" if ladder_start == "Yes" else "(disabled)"))
    if ladder_start_pick is not None:
        r = ladder_start_pick
        st.write(f"{r['Date'].strftime('%Y-%m-%d')}: {r['Team']} vs {r['Opponent']} — {r['BestMarket']} @ {fmt_price(r['BestDecOdds'])} — p={r['BestProb']:.2f}")
        st.write(f"Starter Stake: {ladder_start_amt:.1f} units (already reserved)")
    else:
        st.write("No suitable starter identified.")
with c2:
    st.write("Ladder Continuation " + ("(enabled)" if ladder_cont == "Yes" else "(disabled)"))
    if ladder_cont_pick is not None:
        r = ladder_cont_pick
        st.write(f"{r['Date'].strftime('%Y-%m-%d')}: {r['Team']} vs {r['Opponent']} — {r['BestMarket']} @ {fmt_price(r['BestDecOdds'])} — p={r['BestProb']:.2f}")
        st.write(f"Continuation Stake: {ladder_cont_amt:.1f} units (excluded from slate)")
    else:
        st.write("No suitable continuation identified.")

# =========================================================
# PARLAY BUILDER (80%+ bets + best underdog spreads)
# =========================================================
st.markdown("---")
st.markdown("### Parlay Builder")

# 80%+ bets by model probability (BestProb >= 0.80)
par_cands = pred_df.copy()
par_cands = par_cands[par_cands["BestProb"].notna()]
par_cands = par_cands[par_cands["BestProb"] >= 0.80]
par_cands = par_cands.sort_values(["BestProb","BestEdge"], ascending=[False, False]).reset_index(drop=True)

def leg_text(p):
    m = p["BestMarket"]
    if m in ("Over","Under","OU"):
        side = m if m in ("Over","Under") else ("Over" if (p["OverProb"] and p["OverProb"]>=0.5) else "Under")
        val  = int(p["BestLineVal"]) if p["BestLineVal"] is not None else ""
        return f"{side} {val}".strip()
    if m == "Spread":
        line = p["Line"]
        if line is None:
            return f"{p['Team']} Spread"
        sign = "+" if line > 0 else ""
        return f"{p['Team']} {sign}{line}"
    if m == "Moneyline":
        return f"{p['Team']} ML"
    return "Bet"

def parlay_from_slice(df: pd.DataFrame, max_legs: int = 10):
    legs = df.head(max_legs).to_dict("records")
    if not legs:
        return None
    acc_prob = 1.0
    acc_dec = 1.0
    for leg in legs:
        p = float(leg["BestProb"])
        acc_prob *= p
        dec = leg["BestDecOdds"]
        if not np.isfinite(dec):
            if leg["BestMarket"] in ("Spread","Over","Under","OU"):
                dec = 1.909
            else:
                continue
        acc_dec *= dec
    return legs, acc_prob, acc_dec

if par_cands.empty:
    st.write("No bets with 80%+ model probability yet.")
else:
    # Parlay 1
    slice1 = par_cands.iloc[:10]
    p1 = parlay_from_slice(slice1, max_legs=10)
    if p1 is not None:
        legs1, p_all1, dec_all1 = p1
        st.subheader("80% Bet Parlay 1")
        st.caption(f"{len(legs1)} legs — approx hit probability {p_all1:.2f}, combined odds {decimal_to_american(dec_all1)} ({dec_all1:.2f} dec)")
        rows1 = []
        for j, leg in enumerate(legs1, 1):
            dec = leg["BestDecOdds"]
            if not np.isfinite(dec) and leg["BestMarket"] in ("Spread","Over","Under","OU"):
                dec = 1.909
            rows1.append({
                "Leg": j,
                "Matchup": f"{leg['Team']} vs {leg['Opponent']}",
                "Market": leg_text(leg),
                "Price (US)": decimal_to_american(dec),
                "Model p": f"{leg['BestProb']:.2f}"
            })
        st.dataframe(pd.DataFrame(rows1), use_container_width=True)

    # Parlay 2 (remaining 80%+ bets)
    if len(par_cands) > 10:
        slice2 = par_cands.iloc[10:20]
        p2 = parlay_from_slice(slice2, max_legs=10)
        if p2 is not None:
            legs2, p_all2, dec_all2 = p2
            st.subheader("80% Bet Parlay 2")
            st.caption(f"{len(legs2)} legs — approx hit probability {p_all2:.2f}, combined odds {decimal_to_american(dec_all2)} ({dec_all2:.2f} dec)")
            rows2 = []
            for j, leg in enumerate(legs2, 1):
                dec = leg["BestDecOdds"]
                if not np.isfinite(dec) and leg["BestMarket"] in ("Spread","Over","Under","OU"):
                    dec = 1.909
                rows2.append({
                    "Leg": j,
                    "Matchup": f"{leg['Team']} vs {leg['Opponent']}",
                    "Market": leg_text(leg),
                    "Price (US)": decimal_to_american(dec),
                    "Model p": f"{leg['BestProb']:.2f}"
                })
            st.dataframe(pd.DataFrame(rows2), use_container_width=True)

# Top 5 Underdog ML spreads (most valuable dogs by edge)
st.subheader("Top 5 Underdog ML Spreads")
dog_spreads = pred_df.copy()
dog_spreads = dog_spreads[dog_spreads["IsDogPair"]]
dog_spreads = dog_spreads[dog_spreads["ML"].notna()]
dog_spreads = dog_spreads[dog_spreads["ML"] > 0]  # + odds only
dog_spreads = dog_spreads[dog_spreads["Line"].notna()]
dog_spreads = dog_spreads.sort_values(["BestEdge","CoverProb"], ascending=[False, False]).head(5)

if dog_spreads.empty:
    st.write("No strong underdog ML+spread combos identified yet.")
else:
    rows_d = []
    for _, r in dog_spreads.iterrows():
        sign = "+" if r["Line"] > 0 else ""
        spread_label = f"{r['Team']} {sign}{r['Line']}"
        rows_d.append({
            "Matchup": f"{r['Team']} vs {r['Opponent']}",
            "ML Price": decimal_to_american(american_to_decimal(r["ML"])),
            "Spread": spread_label,
            "Win Prob (ML)": f"{r['WinProb']:.2f}",
            "Cover Prob": f"{r['CoverProb']:.2f}",
            "Edge": f"{r['BestEdge']:.3f}"
        })
    st.dataframe(pd.DataFrame(rows_d), use_container_width=True)

# =========================================================
# DRILLDOWN — show both teams' top-50 ranks (pull opponent row from RAW schedule)
# =========================================================
st.markdown("---")
st.markdown("### Drilldown (click a matchup for team details and Top-50 categories)")

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

for _, r in pred_df.iterrows():
    label = (
        f"{r['Date'].strftime('%b %d, %Y')} — "
        f"{r['Team']} vs {r['Opponent']} — "
        f"Pred {int(round(r['Pred_Points']))}-{int(round(r['Pred_Opp_Points']))}"
    )
    with st.expander(label, expanded=False):
        row_team = r["RowObj"]
        row_opp  = find_opponent_row_raw(r["Team"], r["Opponent"], r["Date"])

        # Meta
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
            st.write(f"**{r['Team']}**")
            st.write(f"Coach: {t_coach}")
            st.write(f"Conference: {t_conf}")
        with cR:
            st.write(f"**{r['Opponent']}**")
            st.write(f"Coach: {o_coach}")
            st.write(f"Conference: {o_conf}")

        st.markdown("---")
        st.write("Top-50 Rank Categories (≤ 50)")

        left_top50 = extract_top50_any(row_team)
        right_top50 = extract_top50_any(row_opp)

        cL2, cR2 = st.columns(2)
        with cL2:
            st.write(f"**{r['Team']}**")
            if not left_top50:
                st.write("None in top 50")
            else:
                st.dataframe(pd.DataFrame([{"Category": k, "Rank": v} for k, v in left_top50]),
                             use_container_width=True)
        with cR2:
            st.write(f"**{r['Opponent']}**")
            if not right_top50:
                st.write("None in top 50")
            else:
                st.dataframe(pd.DataFrame([{"Category": k, "Rank": v} for k, v in right_top50]),
                             use_container_width=True)

# =========================================================
# EXPORT SLATE
# =========================================================
st.markdown("---")
st.markdown("### Export Slate")
export_cols = ["Date","Team","Opponent","BestMarket","BestLineVal","BestDecOdds","BestProb","BestEdge","Stake"]
if not slate.empty:
    export_df = slate[export_cols].copy()
    export_df["Date"] = export_df["Date"].dt.strftime("%Y-%m-%d")
    export_df = export_df.rename(columns={
        "BestLineVal": "MarketVal",
        "BestDecOdds": "DecOdds",
        "BestProb": "BetProb"
    })
    csv = export_df.to_csv(index=False)
    st.download_button("Download Betting Slate CSV", data=csv, file_name="betting_slate.csv", mime="text/csv")
else:
    st.write("No slate to export.")
