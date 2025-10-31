# APP/Pages/9_Betting.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List, Tuple, Optional

from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

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
    # Force MM/DD/YYYY as requested (e.g., 11/03/2025 => Nov 3, 2025)
    return pd.to_datetime(series_like.astype(str).str.strip(), errors="coerce", format="%m/%d/%Y")

def interpret_han(v) -> Optional[str]:
    if pd.isna(v): return None
    s = str(v).strip().upper()
    if s in ("H","HOME"): return "Home"
    if s in ("A","AWAY"): return "Away"
    if s in ("N","NEUTRAL") or "NEUTRAL" in s: return "Neutral"
    return None

def american_to_prob(odds) -> float:
    """American odds -> implied probability (0..1)."""
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    elif o < 0:
        return -o / (-o + 100.0)
    return np.nan

def american_to_decimal(odds) -> float:
    """American -> decimal odds."""
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

def kelly_fraction(prob: float, dec_odds: float, cap: float = 0.05) -> float:
    """Capped Kelly. Returns fraction of bankroll to stake (0..cap)."""
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
    # Treat blanks/NAs as 51+ (i.e., not top-50)
    try:
        v = float(x)
        if np.isnan(v): return 51.0
        return v
    except Exception:
        return 51.0

def extract_top50_from_row(row: pd.Series) -> List[Tuple[str, int]]:
    out = []
    for c in row.index:
        cu = c.upper()
        if "RANK" in cu or cu.endswith(" RANK") or cu.endswith("_RANK"):
            rv = clean_rank_value(row.get(c))
            if rv <= 50:
                out.append((c, int(rv)))
    return sorted(out, key=lambda z: z[1])

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

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
date_col   = find_col(schedule_df, ["Date"])
han_col    = find_col(schedule_df, ["HAN","Home/Away","HomeAway","Location","Loc"])
conf_col   = find_col(schedule_df, ["Conference"])
opp_conf_col = find_col(schedule_df, ["Opponent Conference","Opp Conference"])
coach_col  = find_col(schedule_df, ["Coach Name","Coach","Coach_Name"])
opp_coach_col = find_col(schedule_df, ["Opponent Coach","Opp Coach"])

line_col   = find_col(schedule_df, ["Line"])
ml_col     = find_col(schedule_df, ["ML","Moneyline","Money Line"])
ou_col     = find_col(schedule_df, ["Over/Under Line","OverUnder","Over Under Line","O/U"])

top25_col  = find_col(schedule_df, ["Top 25 Opponent","Top25","Top 25"])
mm_col     = find_col(schedule_df, ["March Madness Opponent","March Madness"])

if team_col is None or opp_col is None or date_col is None:
    st.error("Schedule must include Team, Opponent, and Date.")
    st.stop()

# Parse dates strictly as MM/DD/YYYY
schedule_df["__Date"] = parse_mdy(schedule_df[date_col])
schedule_df = schedule_df.dropna(subset=["__Date"]).copy()

# Deduplicate mirrored fixtures (keep first occurrence; you can improve to prefer 'Home')
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
# BUILD TRAINING DATA (DAILY)
#  - Multi-output regressor: Points, Opp Points
#  - Classifiers: Win, Cover (vs Line), Over (vs OU)
# =========================================================
d_team = find_col(daily_df, ["Team","Teams","team"])
d_opp  = find_col(daily_df, ["Opponent","Opp","opponent"])

d_pts  = find_col(daily_df, ["Points"])
d_opp_pts = find_col(daily_df, ["Opp Points","Opp_Points","OppPoints"])
d_line = find_col(daily_df, ["Line"])
d_ou   = find_col(daily_df, ["Over/Under Line","OverUnder","Over Under Line","O/U"])
d_ml   = find_col(daily_df, ["ML","Moneyline","Money Line"])  # optional, not required for training labels

if d_team is None or d_opp is None or d_pts is None or d_opp_pts is None:
    st.error("Daily predictor must include Team, Opponent, Points, Opp Points.")
    st.stop()

# Features: all numeric columns except targets (Points, Opp Points)
daily_numeric_cols = daily_df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in daily_numeric_cols if c not in (d_pts, d_opp_pts)]

# Categorical columns to encode (present in daily)
cat_cols = []
for c in [d_team, d_opp, "Coach Name", "Coach", "Conference", "Opponent Conference", "Opp Conference", "Opponent Coach"]:
    if c and c in daily_df.columns:
        cat_cols.append(c)
cat_cols = list(dict.fromkeys(cat_cols))

# Assemble X, Y for all tasks
X_num_rows = []
X_cat_rows = []
y_points_rows = []
y_win = []
y_cover = []
y_over  = []

for _, r in daily_df.iterrows():
    t = r.get(d_team)
    o = r.get(d_opp)
    if pd.isna(t) or pd.isna(o):
        continue

    # numeric block
    nums = pd.to_numeric(r[feature_cols], errors="coerce").fillna(0.0).values
    X_num_rows.append(nums)

    # categorical block (string-ify)
    cats = []
    for c in cat_cols:
        cats.append(str(r.get(c)) if c in r.index else "")
    X_cat_rows.append(cats)

    # targets
    pts  = safe_num(r.get(d_pts), default=np.nan)
    oppp = safe_num(r.get(d_opp_pts), default=np.nan)
    if np.isnan(pts) or np.isnan(oppp):
        # we still keep this row for classification if lines/OUs are present?
        # To keep targets aligned across tasks, skip entirely if points missing.
        X_num_rows.pop()
        X_cat_rows.pop()
        continue
    y_points_rows.append([pts, oppp])

    # Win label
    y_win.append(1 if pts > oppp else 0)

    # Cover label (needs line)
    if d_line and d_line in daily_df.columns:
        ln = safe_num(r.get(d_line), default=np.nan)
        # Positive line means the TEAM is favorite by that many points.
        # "Cover" for TEAM means (pts - oppp) - line > 0
        y_cover.append(np.nan if not np.isfinite(ln) else (1 if (pts - oppp - ln) > 0 else 0))
    else:
        y_cover.append(np.nan)

    # Over label (needs OU)
    if d_ou and d_ou in daily_df.columns:
        ouv = safe_num(r.get(d_ou), default=np.nan)
        y_over.append(np.nan if not np.isfinite(ouv) else (1 if (pts + oppp - ouv) > 0 else 0))
    else:
        y_over.append(np.nan)

if len(X_num_rows) == 0:
    st.error("No valid training rows assembled from daily predictor.")
    st.stop()

X_num = np.vstack(X_num_rows)
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_cat = enc.fit_transform(np.array(X_cat_rows, dtype=object))
X_train = np.hstack([X_num, X_cat])

Y_points = np.array(y_points_rows, dtype=float)
y_win = np.array(y_win, dtype=int)

# Filter cover/over labels to valid rows (some may be NaN if no line/ou)
cover_mask = np.array([v in (0,1) for v in y_cover], dtype=bool)
over_mask  = np.array([v in (0,1) for v in y_over], dtype=bool)
y_cover_valid = np.array([int(v) for v in np.array(y_cover, dtype=float)[cover_mask]]) if cover_mask.any() else None
y_over_valid  = np.array([int(v) for v in np.array(y_over, dtype=float)[over_mask]]) if over_mask.any() else None

X_cover = X_train[cover_mask] if cover_mask.any() else None
X_over  = X_train[over_mask]  if over_mask.any()  else None

# =========================================================
# TRAIN MODELS
# =========================================================
st.markdown("### Model Training")
st.caption("Retraining on Daily Predictor each run using all numeric stats + categorical encodings.")

# Multi-output regressor (Points, Opp Points)
rf_points = MultiOutputRegressor(RandomForestRegressor(n_estimators=700, random_state=42))
rf_points.fit(X_train, Y_points)

# Win classifier
clf_win = RandomForestClassifier(n_estimators=600, random_state=42, class_weight="balanced_subsample")
clf_win.fit(X_train, y_win)

# Cover classifier (if labels exist)
clf_cover = None
if X_cover is not None and len(X_cover) >= 20 and len(np.unique(y_cover_valid)) == 2:
    clf_cover = RandomForestClassifier(n_estimators=600, random_state=42, class_weight="balanced_subsample")
    clf_cover.fit(X_cover, y_cover_valid)

# Over classifier (if labels exist)
clf_over = None
if X_over is not None and len(X_over) >= 20 and len(np.unique(y_over_valid)) == 2:
    clf_over = RandomForestClassifier(n_estimators=600, random_state=42, class_weight="balanced_subsample")
    clf_over.fit(X_over, y_over_valid)

# ---------------------------------------------------------
# Quick cross-validated average precision (PR-AUC) snapshot
# ---------------------------------------------------------
def cv_ap(model, X, y, splits=5) -> Optional[float]:
    if X is None or y is None:
        return None
    if len(np.unique(y)) < 2 or X.shape[0] < splits:
        return None
    skf = StratifiedKFold(n_splits=min(splits, X.shape[0]))
    aps = []
    for tr, te in skf.split(X, y):
        m = RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced_subsample")
        m.fit(X[tr], y[tr])
        proba = m.predict_proba(X[te])[:,1]
        aps.append(average_precision_score(y[te], proba))
    return float(np.mean(aps)) if aps else None

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    ap_win = cv_ap(clf_win, X_train, y_win)
    st.write(f"Win (PR-AUC ~): {ap_win:.3f}" if ap_win is not None else "Win (PR-AUC): n/a")
with col_m2:
    ap_cov = cv_ap(clf_cover, X_cover, y_cover_valid) if clf_cover is not None else None
    st.write(f"Cover (PR-AUC ~): {ap_cov:.3f}" if ap_cov is not None else "Cover (PR-AUC): n/a")
with col_m3:
    ap_ov = cv_ap(clf_over, X_over, y_over_valid) if clf_over is not None else None
    st.write(f"Over (PR-AUC ~): {ap_ov:.3f}" if ap_ov is not None else "Over (PR-AUC): n/a")

st.caption("Targets of ~0.70 PR-AUC are goals, not guarantees; improves as Daily grows.")

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
    ladder_cont_amt = st.number_input("Ladder Continuation Units (excluded from base)", min_value=0.0, value=0.0, step=1.0)

colD, colE = st.columns(2)
with colD:
    min_bets, max_bets = st.slider("Bets minimum / maximum", 4, 100, (10, 25))
with colE:
    kelly_cap = st.slider("Max stake per bet (Kelly cap, % of units)", 1, 10, 5) / 100.0

effective_units = max(0.0, units_base - (ladder_cont_amt if ladder_cont == "Yes" else 0.0))
st.info(f"Effective units for slate (excludes ladder continuation): {effective_units:.2f}")

# =========================================================
# BUILD SCHEDULE FEATURE VECTORS (MATCH DAILY'S INPUT SPACE)
# =========================================================
def schedule_row_to_feature(row: pd.Series) -> np.ndarray:
    # Numeric vector in the exact order of feature_cols
    num_vec = []
    for c in feature_cols:
        # Take same-name column if present, else 0.0
        v = row.get(c) if c in row.index else 0.0
        num_vec.append(safe_num(v, 0.0))
    num_vec = np.array(num_vec, dtype=float)

    # Categorical vector in the exact order of cat_cols
    cat_vals = []
    for c in cat_cols:
        # Try to pull from schedule with the same or analogous name
        if c in row.index:
            cat_vals.append(str(row.get(c)))
        elif c == d_team and team_col in row.index:
            cat_vals.append(str(row.get(team_col)))
        elif c == d_opp and opp_col in row.index:
            cat_vals.append(str(row.get(opp_col)))
        elif "Opponent" in c and opp_col in row.index:
            cat_vals.append(str(row.get(opp_col)))
        elif c in ("Coach Name","Coach","Coach_Name") and coach_col in row.index:
            cat_vals.append(str(row.get(coach_col)))
        elif c in ("Opponent Coach","Opp Coach") and opp_coach_col in row.index:
            cat_vals.append(str(row.get(opp_coach_col)))
        elif c in ("Conference","Conf") and conf_col in row.index:
            cat_vals.append(str(row.get(conf_col)))
        elif c in ("Opponent Conference","Opp Conference") and opp_conf_col in row.index:
            cat_vals.append(str(row.get(opp_conf_col)))
        else:
            cat_vals.append("")
    cat_arr = enc.transform([cat_vals])  # shape (1, k)
    x = np.hstack([num_vec, cat_arr.ravel()])

    # Align width to X_train
    if x.shape[0] != X_train.shape[1]:
        tmp = np.zeros((X_train.shape[1],), dtype=float)
        m = min(len(tmp), len(x))
        tmp[:m] = x[:m]
        x = tmp
    return x

# =========================================================
# SCORE EACH SCHEDULED GAME
#  - Predict Points/Opp Points, Win prob
#  - Predict Cover prob (TEAM vs its Line), Over prob (vs OU)
#  - Price edges with market odds (-110 fallback for spread/total)
# =========================================================
rows = []
for _, r in schedule_df.iterrows():
    t = str(r[team_col]).strip()
    o = str(r[opp_col]).strip()

    x = schedule_row_to_feature(r)
    pts, oppp = rf_points.predict(x.reshape(1, -1))[0]
    pts   = max(0.0, float(pts))
    oppp  = max(0.0, float(oppp))
    margin = pts - oppp
    total  = pts + oppp

    # Win probability (TEAM perspective)
    win_p = float(clf_win.predict_proba(x.reshape(1, -1))[0][1])

    # Market terms
    line = safe_num(r.get(line_col), default=np.nan) if line_col else np.nan
    ouv  = safe_num(r.get(ou_col), default=np.nan) if ou_col else np.nan
    mline = safe_num(r.get(ml_col), default=np.nan) if ml_col else np.nan

    # Spread cover probability for TEAM (if classifier exists & schedule has line)
    cover_prob = np.nan
    if clf_cover is not None and np.isfinite(line):
        cover_prob = float(clf_cover.predict_proba(x.reshape(1, -1))[0][1])
    elif np.isfinite(line):
        # fallback logistic proxy on margin vs line (conservative)
        sigma = 8.5
        cover_prob = sigmoid((margin - line) / sigma)

    # Over probability (if classifier exists & schedule has OU)
    over_prob = np.nan
    if clf_over is not None and np.isfinite(ouv):
        over_prob = float(clf_over.predict_proba(x.reshape(1, -1))[0][1])
    elif np.isfinite(ouv):
        tau = 12.0
        over_prob = sigmoid((total - ouv) / tau)

    # Moneyline
    ml_dec = american_to_decimal(mline) if np.isfinite(mline) else np.nan
    ml_edge = np.nan
    ml_kelly = 0.0
    if np.isfinite(ml_dec):
        ml_edge = win_p - (1.0 / ml_dec)
        ml_kelly = kelly_fraction(win_p, ml_dec, cap=kelly_cap)

    # Spread pricing: assume -110 (-110 both ways) if no explicit price
    spread_dec = 1.909 if np.isfinite(line) else np.nan
    spread_edge = (cover_prob - (1.0 / spread_dec)) if (np.isfinite(spread_dec) and np.isfinite(cover_prob)) else np.nan
    spread_kelly = kelly_fraction(cover_prob, spread_dec, cap=kelly_cap) if np.isfinite(spread_edge) else 0.0

    # Totals pricing (Over and Under at -110)
    ou_dec = 1.909 if np.isfinite(ouv) else np.nan
    if np.isfinite(over_prob) and np.isfinite(ou_dec):
        # pick best side by edge
        over_edge  = over_prob - (1.0 / ou_dec)
        under_prob = 1.0 - over_prob
        under_edge = under_prob - (1.0 / ou_dec)
        if over_edge >= under_edge:
            ou_pick, ou_prob, ou_edge_v = "Over", over_prob, over_edge
        else:
            ou_pick, ou_prob, ou_edge_v = "Under", under_prob, under_edge
        ou_kelly = kelly_fraction(ou_prob, ou_dec, cap=kelly_cap)
    else:
        ou_pick, ou_prob, ou_edge_v, ou_kelly = None, np.nan, np.nan, 0.0

    # Priority tiers (Top25 > MM > Power5)
    is_top25 = False
    if top25_col and top25_col in schedule_df.columns:
        v = r.get(top25_col)
        is_top25 = (str(v).strip() not in ("0","", "N","FALSE","NaN"))

    is_mm = False
    if mm_col and mm_col in schedule_df.columns:
        v = r.get(mm_col)
        is_mm = (str(v).strip() not in ("0","", "N","FALSE","NaN"))

    conf = r.get(opp_conf_col) if opp_conf_col in r.index else r.get(conf_col)
    cc = str(conf).strip().upper() if conf is not None else ""
    is_p5 = cc in ("SEC","BIG TEN","B1G","BIG 12","ACC")

    # Choose best “single” bet by edge
    edges = []
    if np.isfinite(spread_edge):
        edges.append(("Spread", spread_edge, cover_prob, spread_dec, spread_kelly, line))
    if np.isfinite(ml_edge):
        edges.append(("Moneyline", ml_edge, win_p, ml_dec, ml_kelly, mline))
    if np.isfinite(ou_edge_v):
        edges.append((ou_pick or "OU", ou_edge_v, ou_prob, ou_dec, ou_kelly, ouv))
    best = max(edges, key=lambda z: (z[1] if np.isfinite(z[1]) else -1e9), default=None)

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
        "RowObj": r
    })

pred_df = pd.DataFrame(rows)
if pred_df.empty:
    st.info("No priced games available.")
    st.stop()

# =========================================================
# LADDER PICKS (most confident, odds near even: -150..+150)
# =========================================================
def pick_ladder(df: pd.DataFrame, exclude_key=None):
    cands = []
    for _, r in df.iterrows():
        mkt = r["BestMarket"]
        if not mkt: continue
        key = (r["Team"], r["Opponent"], r["Date"], mkt)
        if exclude_key and key == exclude_key:
            continue
        dec = r["BestDecOdds"]
        if not np.isfinite(dec):
            # assume -110 for spread/OU
            if mkt in ("Spread","Over","Under","OU"):
                dec = 1.909
            else:
                continue
        # keep bets around even odds
        if 1.67 <= dec <= 2.5:  # ~ -150 to +150
            cands.append((r["BestProb"], -abs(dec-2.0), r))
    if not cands:
        return None
    cands.sort(key=lambda z: (z[0], z[1]), reverse=True)
    return cands[0][2]

# =========================================================
# SLATE CONSTRUCTION & STAKING
# =========================================================
# Priority sort: Top25 → MM → Power5 → Date, then by BestEdge desc
pred_df["_prio"] = pred_df.apply(
    lambda r: (0 if r["IsTop25"] else 1, 0 if r["IsMM"] else 1, 0 if r["IsP5"] else 1, r["Date"]),
    axis=1
)
pred_df = pred_df.sort_values(by=["_prio","BestEdge"], ascending=[True, False]).reset_index(drop=True)

# Build slate from positive-edge bets
slate = pred_df.dropna(subset=["BestEdge"]).copy()
slate = slate[slate["BestEdge"] > 0].copy()
slate = slate.sort_values(by=["_prio","BestEdge"], ascending=[True, False]).reset_index(drop=True)

N = min(max_bets, max(min_bets, len(slate)))
slate = slate.head(N).copy()

def stake_for_row(r) -> float:
    f = float(r["BestKelly"])
    if not np.isfinite(f) or f <= 0:
        return 0.0
    return round(f * effective_units, 2)

slate["Stake"] = slate.apply(stake_for_row, axis=1)

# If unpriced => all zero stakes; fallback to flat ~15% spread
if slate["Stake"].sum() == 0 and len(slate) > 0:
    target_spend = 0.15 * effective_units
    per = round(target_spend / len(slate), 2)
    slate["Stake"] = per

# Ladder selection
ladder_start_pick, ladder_cont_pick = None, None
ex_key = None
if ladder_start == "Yes":
    ladder_start_pick = pick_ladder(pred_df)
    if ladder_start_pick is not None:
        ex_key = (ladder_start_pick["Team"], ladder_start_pick["Opponent"], ladder_start_pick["Date"], ladder_start_pick["BestMarket"])

if ladder_cont == "Yes":
    ladder_cont_pick = pick_ladder(pred_df, exclude_key=ex_key)

# =========================================================
# PRESENTATION
# =========================================================
st.markdown("---")
st.markdown("### Suggested Slate (Model-Driven)")

def fmt_price(dec):
    if not np.isfinite(dec):
        return "-110"  # default for spread/OU display when not explicit
    return decimal_to_american(dec)

if slate.empty:
    st.info("No +EV bets under current markets.")
else:
    view = slate[[
        "Date","Team","Opponent","BestMarket","BestLineVal",
        "Pred_Margin","Pred_Total","WinProb","BestProb","BestEdge","BestDecOdds","Stake"
    ]].copy()
    view = view.rename(columns={
        "BestLineVal": "Market",
        "BestDecOdds": "Price (dec)",
        "BestProb": "Bet Prob",
        "WinProb": "ML Prob"
    })
    view["Date"] = view["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(view, use_container_width=True)

# Example text summary of stake plan (like the format you showed)
if not slate.empty:
    st.markdown("**Stake Plan (example wording)**")
    lines = []
    # Pick some compositions reminiscent of your example, without forcing exact counts
    for _, r in slate.iterrows():
        market = r["BestMarket"]
        if market in ("Over","Under","OU"):
            pick_name = f"{market} {int(r['BestLineVal']) if r['BestLineVal'] is not None else ''}".strip()
        elif market == "Spread":
            sign = "+" if (r['BestLineVal'] is not None and r['BestLineVal'] > 0) else ""
            pick_name = f"{r['Team']} {sign}{r['BestLineVal']}" if r['BestLineVal'] is not None else f"{r['Team']} Spread"
        elif market == "Moneyline":
            pick_name = f"{r['Team']} ML"
        else:
            pick_name = market or "Bet"

        lines.append(f"{int(round(r['Stake']))} on {pick_name}")
    st.write("; ".join(lines))

st.markdown("---")
st.markdown("### Ladders")
c1, c2 = st.columns(2)
with c1:
    st.write("Ladder Starter " + ("(enabled)" if ladder_start == "Yes" else "(disabled)"))
    if ladder_start_pick is not None:
        r = ladder_start_pick
        st.write(f"{r['Date'].strftime('%Y-%m-%d')}: {r['Team']} vs {r['Opponent']} — {r['BestMarket']} @ {fmt_price(r['BestDecOdds'])} — p={r['BestProb']:.2f}")
    else:
        st.write("No suitable starter identified.")

with c2:
    st.write("Ladder Continuation " + ("(enabled)" if ladder_cont == "Yes" else "(disabled)"))
    if ladder_cont_pick is not None:
        r = ladder_cont_pick
        st.write(f"{r['Date'].strftime('%Y-%m-%d')}: {r['Team']} vs {r['Opponent']} — {r['BestMarket']} @ {fmt_price(r['BestDecOdds'])} — p={r['BestProb']:.2f}")
        st.write(f"Continuation Stake: {ladder_cont_amt:.2f} units (excluded from slate units)")
    else:
        st.write("No suitable continuation identified.")

# =========================================================
# DRILLDOWN: Team details + Top-50 category splits
# =========================================================
st.markdown("---")
st.markdown("### Drilldown (click a matchup for team details and Top-50 categories)")

for _, r in pred_df.iterrows():
    label = (
        f"{r['Date'].strftime('%b %d, %Y')} — "
        f"{r['Team']} vs {r['Opponent']} — "
        f"Pred {int(round(r['Pred_Points']))}-{int(round(r['Pred_Opp_Points']))}"
    )
    with st.expander(label, expanded=False):
        rowobj = r["RowObj"]

        t_conf = rowobj.get(conf_col) if conf_col in rowobj.index else ""
        o_conf = rowobj.get(opp_conf_col) if opp_conf_col in rowobj.index else ""
        t_coach = rowobj.get(coach_col) if coach_col in rowobj.index else ""
        o_coach = rowobj.get(opp_coach_col) if opp_coach_col in rowobj.index else ""

        # Current record if available
        t_wins = rowobj.get("Wins") if "Wins" in rowobj.index else None
        t_losses = rowobj.get("Losses") if "Losses" in rowobj.index else None

        cL, cR = st.columns(2)
        with cL:
            st.write(f"**{r['Team']}**")
            st.write(f"Coach: {t_coach}")
            st.write(f"Conference: {t_conf}")
            if t_wins is not None:
                st.write(f"Record: {t_wins}-{t_losses if t_losses is not None else ''}")

        with cR:
            st.write(f"**{r['Opponent']}**")
            st.write(f"Coach: {o_coach}")
            st.write(f"Conference: {o_conf}")

        st.markdown("---")
        st.write("Top-50 Rank Categories")

        # Left team (row perspective)
        left_top50 = extract_top50_from_row(rowobj)
        cL2, cR2 = st.columns(2)
        with cL2:
            st.write(f"**{r['Team']}**")
            if not left_top50:
                st.write("None in top 50")
            else:
                st.dataframe(pd.DataFrame([{"Category": k, "Rank": v} for k, v in left_top50]),
                             use_container_width=True)

        # Opponent: heuristic — pull any OPP_* rank columns <= 50
        opp_top50 = []
        for k in rowobj.index:
            ku = k.upper()
            if "OPP_" in ku and ("RANK" in ku or ku.endswith(" RANK") or ku.endswith("_RANK")):
                rv = clean_rank_value(rowobj.get(k))
                if rv <= 50:
                    opp_top50.append((k, int(rv)))
        opp_top50 = sorted(opp_top50, key=lambda z: z[1])
        with cR2:
            st.write(f"**{r['Opponent']}**")
            if not opp_top50:
                st.write("None in top 50")
            else:
                st.dataframe(pd.DataFrame([{"Category": k, "Rank": v} for k, v in opp_top50]),
                             use_container_width=True)

# =========================================================
# PARLAY IDEAS (from highest probability singles)
# =========================================================
st.markdown("---")
st.markdown("### Parlay Ideas")
parlay_source = pred_df.sort_values(["BestProb","BestEdge"], ascending=[False, False]).head(20).copy()

# Map homerun slider (1..10) → legs (2..5)
min_legs = int(np.interp(homerun, [1, 10], [2, 5]))
max_legs = min_legs + 1

def build_parlays(df: pd.DataFrame, min_legs: int, max_legs: int, max_sets: int = 3):
    out = []
    picks = df.head(12).to_dict("records")
    used = set()
    for L in range(min_legs, max_legs + 1):
        if len(out) >= max_sets:
            break
        legs, acc_prob, acc_dec = [], 1.0, 1.0
        for p in picks:
            key = (p["Team"], p["Opponent"], p["Date"], p["BestMarket"])
            if key in used: continue
            if not np.isfinite(p["BestProb"]): continue
            dec = p["BestDecOdds"]
            if not np.isfinite(dec):
                dec = 1.909 if p["BestMarket"] in ("Spread","Over","Under","OU") else np.nan
            if not np.isfinite(dec): continue
            legs.append(p)
            acc_prob *= p["BestProb"]
            acc_dec  *= dec
            used.add(key)
            if len(legs) >= L: break
        if len(legs) == L and acc_prob > 0:
            out.append((legs, acc_prob, acc_dec))
    return out

parlays = build_parlays(parlay_source, min_legs, max_legs, max_sets=3)
if not parlays:
    st.write("No parlay suggestions currently.")
else:
    for i, (legs, p_all, dec_all) in enumerate(parlays, 1):
        st.write(f"Parlay {i}: {len(legs)} legs — price {dec_all:.2f} (~{decimal_to_american(dec_all)}) — hit prob ~{p_all:.2f}")
        for leg in legs:
            st.write(f"• {leg['Team']} vs {leg['Opponent']} — {leg['BestMarket']} @ {decimal_to_american(leg['BestDecOdds'])} (p={leg['BestProb']:.2f})")

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
