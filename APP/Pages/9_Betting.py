# APP/Pages/9_Betting.py
# =========================================================
# Betting Page — full Streamlit app with upgraded ML pipeline
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
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
    # Force MM/DD/YYYY as requested (e.g., 11/03/2025)
    # Also allow common alternates to be robust (coerce invalid)
    s = series_like.astype(str).str.strip()
    out = pd.to_datetime(s, errors="coerce", format="%m/%d/%Y")
    # if still NaT, try some common formats found in Daily
    mask = out.isna()
    if mask.any():
        try2 = pd.to_datetime(s[mask], errors="coerce", infer_datetime_format=True)
        out.loc[mask] = try2
    return out

def interpret_han(v) -> Optional[str]:
    if pd.isna(v): return None
    s = str(v).strip().upper()
    if s in ("H","HOME"): return "Home"
    if s in ("A","AWAY"): return "Away"
    if s in ("N","NEUTRAL") or "NEUTRAL" in s: return "Neutral"
    return None

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

def normal_cdf(x):
    return 0.5 * (1.0 + np.math.erf(x / np.sqrt(2.0)))

def uniq_jitter(key: str, scale: float = 0.07) -> float:
    """Deterministic jitter so predictions aren't identical."""
    h = abs(hash(key)) % 10_000
    return (h / 10_000.0 - 0.5) * 2.0 * scale  # [-scale, +scale]

def zero_as_missing_mask(df: pd.DataFrame, numeric_cols: List[str], frac_threshold: float = 0.6) -> List[str]:
    """
    Heuristic: if a numeric column has a very high fraction of zeros,
    it's likely 0 means 'missing' in your exports. We'll treat zeros as NaN for those columns.
    """
    cols = []
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        # Only consider if there are non-nulls
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

# Parse dates strictly as MM/DD/YYYY (with fallback)
schedule_df["__Date"] = parse_mdy(schedule_df[date_col])
schedule_df = schedule_df.dropna(subset=["__Date"]).copy()

# Deduplicate mirrored fixtures (keep first occurrence)
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
# BUILD TRAINING DATA (DAILY) — FEATURE SPACE
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

# Numeric columns (include objects that are numeric)
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

# Remove target columns from features
for drop_c in [d_pts, d_opp_pts]:
    if drop_c in daily_numeric_cols:
        daily_numeric_cols.remove(drop_c)

# Categorical columns (teams, conferences, coaches, HAN/location)
cat_candidates = [
    d_team, d_opp,
    "Coach Name","Coach","Coach_Name",
    "Opponent Coach","Opp Coach",
    "Conference","Opponent Conference","Opp Conference",
    "HAN","Home/Away","HomeAway","Location","Loc"
]
cat_cols = [c for c in cat_candidates if c and c in daily_df.columns]
cat_cols = list(dict.fromkeys(cat_cols))

# Treat zeros as missing when columns are zero-dominant
zero_missing_cols = zero_as_missing_mask(daily_df, daily_numeric_cols, frac_threshold=0.6)
daily_df_num = impute_numeric_with_zero_missing(daily_df, daily_numeric_cols, zero_missing_cols)

# Build train frame with valid target rows
mask_targets = pd.to_numeric(daily_df[d_pts], errors="coerce").notna() & pd.to_numeric(daily_df[d_opp_pts], errors="coerce").notna()
df_train = daily_df_num.loc[mask_targets].copy()

Y_points = df_train[[d_pts, d_opp_pts]].apply(pd.to_numeric, errors="coerce").values

# Preprocessor
num_transformer = SimpleImputer(strategy="median")
cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

preproc = ColumnTransformer(
    transformers=[
        ("num", num_transformer, daily_numeric_cols),
        ("cat", cat_transformer, cat_cols)
    ],
    remainder="drop",
    sparse_threshold=0.0
)

X_train = preproc.fit_transform(df_train)

# Labels for win
y_win = (df_train[d_pts].astype(float) > df_train[d_opp_pts].astype(float)).astype(int).values

# =========================================================
# TRAIN MODELS
# =========================================================
st.markdown("### Model Training")
st.caption("OneHot for team/conference/coach/HAN; numeric median-impute with 0→NaN heuristic for sparse columns.")

# Points model (multi-output RF)
rf_points = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=800, max_depth=None, random_state=42, n_jobs=-1
))
rf_points.fit(X_train, Y_points)

# Win classifier
clf_win = RandomForestClassifier(
    n_estimators=700, random_state=42, class_weight="balanced_subsample", n_jobs=-1
)
clf_win.fit(X_train, y_win)

# Residual spreads for margin & total (used to convert to P(cover) & P(over))
train_pred_pts = rf_points.predict(X_train)
train_margin   = train_pred_pts[:,0] - train_pred_pts[:,1]
true_margin    = Y_points[:,0] - Y_points[:,1]
train_total    = train_pred_pts.sum(axis=1)
true_total     = Y_points.sum(axis=1)

res_margin = (true_margin - train_margin)
res_total  = (true_total  - train_total)

sigma_margin = max(5.5, float(np.nanstd(res_margin, ddof=1)) * 0.9)
sigma_total  = max(9.0, float(np.nanstd(res_total,  ddof=1)) * 0.9)

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
        proba = m.predict_proba(X[te])[:,1]
        aps.append(average_precision_score(y[te], proba))
    return float(np.mean(aps)) if aps else None

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    ap_win = cv_ap_class(X_train, y_win)
    st.write(f"Win (PR-AUC ~): {ap_win:.3f}" if ap_win is not None else "Win (PR-AUC): n/a")
with col_m2:
    st.write(f"σ_margin ≈ {sigma_margin:.2f}")
with col_m3:
    st.write(f"σ_total ≈ {sigma_total:.2f}")

st.caption("Cover/Over probabilities are derived from learned (margin,total) distributions vs Line/OU.")

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
# BUILD SCHEDULE FEATURE VECTORS (MATCH PREPROC)
# =========================================================
def schedule_row_to_feature(row: pd.Series) -> np.ndarray:
    """
    Build a single-row dataframe with the same schema names and pass through `preproc`.
    Any missing numeric features default to NaN (imputed later).
    """
    data = {}
    # Numerics
    for c in daily_numeric_cols:
        data[c] = safe_num(row.get(c), np.nan)
    # Categoricals by name mapping
    data_map = {
        d_team: team_col,
        d_opp: opp_col,
        "Coach Name": coach_col, "Coach": coach_col, "Coach_Name": coach_col,
        "Opponent Coach": opp_coach_col, "Opp Coach": opp_coach_col,
        "Conference": conf_col,
        "Opponent Conference": opp_conf_col, "Opp Conference": opp_conf_col,
        "HAN": han_col, "Home/Away": han_col, "HomeAway": han_col,
        "Location": han_col, "Loc": han_col
    }
    for c in cat_cols:
        src = data_map.get(c, None)
        if src and src in row.index:
            data[c] = str(row.get(src))
        else:
            if c == d_team and team_col in row.index: data[c] = str(row.get(team_col))
            elif c == d_opp and opp_col in row.index: data[c] = str(row.get(opp_col))
            else: data[c] = ""

    one = pd.DataFrame([data], columns=list(dict.fromkeys(daily_numeric_cols + cat_cols)))
    # Apply the same zero→NaN heuristic for schedule rows:
    for c in zero_missing_cols:
        if c in one.columns:
            v = pd.to_numeric(one.loc[0, c], errors="coerce")
            if v == 0: one.loc[0, c] = np.nan
    X = preproc.transform(one)
    return X

# =========================================================
# SCORE EACH SCHEDULED GAME (Points → Margin/Total → Probs)
# =========================================================
rows = []
for _, r in schedule_df.iterrows():
    t = str(r[team_col]).strip()
    o = str(r[opp_col]).strip()

    X = schedule_row_to_feature(r)
    pts_pred = rf_points.predict(X)[0]
    pts, oppp = float(pts_pred[0]), float(pts_pred[1])

    # Deterministic small jitter to avoid identical scores
    jitter = uniq_jitter(f"{t}|{o}|{r['__Date']}")
    pts  = max(0.0, pts  + jitter)
    oppp = max(0.0, oppp - jitter)

    margin = pts - oppp
    total  = pts + oppp

    # Win prob from margin distribution
    win_p = normal_cdf(margin / max(1e-6, sigma_margin))
    win_p = min(0.995, max(0.005, win_p))

    # Market terms
    line = safe_num(r.get(line_col), default=np.nan) if line_col else np.nan
    ouv  = safe_num(r.get(ou_col),   default=np.nan) if ou_col   else np.nan
    mline= safe_num(r.get(ml_col),   default=np.nan) if ml_col   else np.nan

    # Cover prob = P(margin > line)
    cover_prob = np.nan
    if np.isfinite(line):
        cover_prob = normal_cdf((margin - line) / max(1e-6, sigma_margin))
        cover_prob = min(0.995, max(0.005, cover_prob))

    # Over prob = P(total > OU)
    over_prob = np.nan
    if np.isfinite(ouv):
        over_prob = normal_cdf((total - ouv) / max(1e-6, sigma_total))
        over_prob = min(0.995, max(0.005, over_prob))

    # Moneyline odds → edge & Kelly
    ml_dec = american_to_decimal(mline) if np.isfinite(mline) else np.nan
    ml_edge = np.nan
    ml_kelly = 0.0
    if np.isfinite(ml_dec):
        ml_edge  = win_p - (1.0 / ml_dec)
        ml_kelly = kelly_fraction(win_p, ml_dec, cap=kelly_cap)

    # Spread pricing: assume -110 if not given
    spread_dec = 1.909 if np.isfinite(line) else np.nan
    spread_edge = (cover_prob - (1.0 / spread_dec)) if (np.isfinite(spread_dec) and np.isfinite(cover_prob)) else np.nan
    spread_kelly = kelly_fraction(cover_prob, spread_dec, cap=kelly_cap) if np.isfinite(spread_edge) else 0.0

    # Totals pricing (choose Over/Under side with higher edge)
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
        ou_kelly = kelly_fraction(ou_prob, ou_dec, cap=kelly_cap)

    # Priority tiers
    is_top25 = False
    if top25_col and top25_col in schedule_df.columns:
        v = r.get(top25_col); is_top25 = (str(v).strip() not in ("0","","N","FALSE","NaN"))

    is_mm = False
    if mm_col and mm_col in schedule_df.columns:
        v = r.get(mm_col); is_mm = (str(v).strip() not in ("0","","N","FALSE","NaN"))

    conf = r.get(opp_conf_col) if opp_conf_col in r.index else r.get(conf_col)
    cc = str(conf).strip().upper() if conf is not None else ""
    is_p5 = cc in ("SEC","BIG TEN","B1G","BIG 12","ACC","PAC-12","PAC 12")

    # Choose best single bet by edge
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

# Flat fallback if Kelly returns zeros
if slate["Stake"].sum() == 0 and len(slate) > 0:
    target_spend = 0.15 * effective_units
    per = round(target_spend / len(slate), 2)
    slate["Stake"] = per

# =========================================================
# PRESENTATION — CLEAN TABLE + REMAINDER LOGIC
# =========================================================
st.markdown("---")
st.markdown("### Suggested Slate (Model-Driven)")

def pick_label(r):
    if r["BestMarket"] in ("Over","Under","OU"):
        side = r["BestMarket"] if r["BestMarket"] in ("Over","Under") else ("Over" if (r["OverProb"] and r["OverProb"]>=0.5) else "Under")
        val = int(r["BestLineVal"]) if r["BestLineVal"] is not None else ""
        return f"{side} {val}".strip()
    elif r["BestMarket"] == "Spread":
        sign = "+" if (r['BestLineVal'] is not None and r['BestLineVal'] > 0) else ""
        return f"{r['Team']} {sign}{r['BestLineVal']}" if r['BestLineVal'] is not None else f"{r['Team']} Spread"
    elif r["BestMarket"] == "Moneyline":
        return f"{r['Team']} ML"
    return r["BestMarket"] or "Bet"

if slate.empty:
    st.info("No +EV bets under current markets.")
else:
    # Compute remainder against desired effective_units spend
    spent_units = float(slate["Stake"].sum())
    remainder = round(max(0.0, effective_units - spent_units), 2)

    plan_rows = [{
        "GAME": f"{r['Team']} vs {r['Opponent']}",
        "BET TYPE": pick_label(r),
        "BET AMOUNT": f"{round(r['Stake'])} UNITS"
    } for _, r in slate.iterrows()]

    # Assign remainder to Most probable ML if needed
    if remainder > 0:
        ml_cand = pred_df[pred_df["BestMarket"]=="Moneyline"].copy()
        ml_cand = ml_cand.sort_values(["BestProb","BestEdge"], ascending=[False, False])
        if not ml_cand.empty:
            top = ml_cand.iloc[0]
            plan_rows.append({
                "GAME": f"{top['Team']} vs {top['Opponent']}",
                "BET TYPE": "Most probable ML (remainder)",
                "BET AMOUNT": f"{remainder} UNITS"
            })
            st.info("Remainder assigned to the most probable ML — end of smart bets.")

    plan_df = pd.DataFrame(plan_rows, columns=["GAME","BET TYPE","BET AMOUNT"])
    st.dataframe(plan_df, use_container_width=True)

# =========================================================
# LADDERS
# =========================================================
st.markdown("---")
st.markdown("### Ladders")
c1, c2 = st.columns(2)

def fmt_price(dec):
    if not np.isfinite(dec):
        return "-110"
    return decimal_to_american(dec)

ladder_start_pick, ladder_cont_pick = None, None
ex_key = None
if ladder_start == "Yes":
    ladder_start_pick = pick_ladder(pred_df)
    if ladder_start_pick is not None:
        ex_key = (ladder_start_pick["Team"], ladder_start_pick["Opponent"], ladder_start_pick["Date"], ladder_start_pick["BestMarket"])

if ladder_cont == "Yes":
    ladder_cont_pick = pick_ladder(pred_df, exclude_key=ex_key)

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
