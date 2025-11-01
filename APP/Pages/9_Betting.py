# APP/Pages/9_Betting.py
# =========================================================
# Betting Page — Predict Points & Opp Points only; derive everything else
# Diversified staking (no per-bet cap), robust features, and cleaner displays
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import List, Tuple, Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, StratifiedKFold
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
    s = series_like.astype(str).str.strip()
    out = pd.to_datetime(s, errors="coerce", format="%m/%d/%Y")
    if out.isna().any():
        m = out.isna()
        out.loc[m] = pd.to_datetime(s[m], errors="coerce", infer_datetime_format=True)
    return out

def american_to_decimal(odds) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0: return 1.0 + (o / 100.0)
    if o < 0: return 1.0 + (100.0 / -o)
    return np.nan

def decimal_to_american(dec: float) -> str:
    if not np.isfinite(dec) or dec <= 1.0: return "N/A"
    if dec >= 2.0: return f"+{int(round((dec - 1.0) * 100))}"
    return f"{int(round(-100.0 / (dec - 1.0)))}"

def normal_cdf(x):
    return 0.5 * (1.0 + np.math.erf(x / np.sqrt(2.0)))

def kelly_fraction(prob: float, dec_odds: float, cap: float = 1.0) -> float:
    # cap=1.0 effectively removes the per-bet cap; we normalize later to diversify
    if not np.isfinite(prob) or not np.isfinite(dec_odds) or dec_odds <= 1.0: return 0.0
    p = max(0.0, min(1.0, prob))
    b = dec_odds - 1.0
    if b <= 0: return 0.0
    f = (p * (b + 1.0) - 1.0) / b
    return float(max(0.0, f))

def safe_num(x, default=np.nan) -> float:
    try:
        v = float(x)
        if np.isnan(v): return default
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

def extract_top50_from_row(row: pd.Series) -> List[Tuple[str, int]]:
    out = []
    for c in row.index:
        cu = c.upper()
        if "RANK" in cu or cu.endswith(" RANK") or cu.endswith("_RANK"):
            rv = clean_rank_value(row.get(c))
            if rv <= 50:
                out.append((c, int(rv)))
    return sorted(out, key=lambda z: z[1])

def uniq_jitter(key: str, scale: float = 0.07) -> float:
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

def fmt_price(dec):
    if not np.isfinite(dec): return "-110"
    return decimal_to_american(dec)

# =========================================================
# LOAD
# =========================================================
schedule_df = load_csv(SCHEDULE_FILE)
daily_df    = load_csv(DAILY_FILE)
if schedule_df is None or daily_df is None:
    st.stop()

# =========================================================
# COLUMN DETECTION
# =========================================================
team_col   = find_col(schedule_df, ["Team","Teams"])
opp_col    = find_col(schedule_df, ["Opponent","Opp","opponent"])
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

schedule_df["__Date"] = parse_mdy(schedule_df[date_col])
schedule_df = schedule_df.dropna(subset=["__Date"]).copy()

# Deduplicate mirrored fixtures (keep first occurrence)
seen, keep_idx = set(), []
for idx, r in schedule_df.iterrows():
    t = str(r[team_col]).strip()
    o = str(r[opp_col]).strip()
    d = r["__Date"].date()
    key = tuple(sorted([t.lower(), o.lower()])) + (d,)
    if key in seen: continue
    seen.add(key); keep_idx.append(idx)
schedule_df = schedule_df.loc[keep_idx].reset_index(drop=True)

# =========================================================
# TRAIN DATA (DAILY) — Predict Points & Opp Points ONLY
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

# --- pick informative numeric features ---
exclude_labels = {d_pts, d_opp_pts, "SM", "Total Points", "Total", "Opp Points"}
exclude_text   = {"Date","Location","Loc","Home/Away","HomeAway","HAN","Coach Name","Coach","Coach_Name",
                  "Opponent Coach","Opp Coach","Conference","Opponent Conference","Opp Conference"}
candidate_cols = [c for c in daily_df.columns if c not in exclude_labels]

# numeric detection: keep columns with at least 20% numeric (non-NaN after coercion)
daily_numeric_cols = []
for c in candidate_cols:
    if c in exclude_text: 
        continue
    s = pd.to_numeric(daily_df[c], errors="coerce")
    if (s.notna().mean() >= 0.2):
        daily_numeric_cols.append(c)

# categorical columns we want one-hot
cat_candidates = [
    d_team, d_opp,
    "Coach Name","Coach","Coach_Name",
    "Opponent Coach","Opp Coach",
    "Conference","Opponent Conference","Opp Conference",
    "HAN","Home/Away","HomeAway","Location","Loc"
]
cat_cols = [c for c in cat_candidates if c and c in daily_df.columns]
cat_cols = list(dict.fromkeys(cat_cols))

# zero→NaN heuristic, numeric coercion
zero_missing_cols = zero_as_missing_mask(daily_df, daily_numeric_cols, frac_threshold=0.6)
daily_df_num = impute_numeric_with_zero_missing(daily_df, daily_numeric_cols, zero_missing_cols)

# only rows with valid targets
mask_targets = pd.to_numeric(daily_df[d_pts], errors="coerce").notna() & pd.to_numeric(daily_df[d_opp_pts], errors="coerce").notna()
df_train = daily_df_num.loc[mask_targets].copy()
Y_points = df_train[[d_pts, d_opp_pts]].apply(pd.to_numeric, errors="coerce").values

# Preprocessor
num_transformer = SimpleImputer(strategy="median")
# scikit-learn compatibility: 1.2+ uses sparse_output
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

X_train = preproc.fit_transform(df_train)

# =========================================================
# MODELS — Points & Opp Points regressor, metrics via derived labels
# =========================================================
st.markdown("### Model Training")
st.caption("Numerics dominate; Teams/Confs/Coaches/HAN one-hot; zeros→NaN on sparse columns; targets = Points & Opp Points.")

rf_points = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=1200, max_depth=None, max_features="sqrt", random_state=42, n_jobs=-1
    )
)
rf_points.fit(X_train, Y_points)

# Estimate residual spreads for margin/total
train_pred_pts = rf_points.predict(X_train)
train_margin   = train_pred_pts[:,0] - train_pred_pts[:,1]
true_margin    = Y_points[:,0] - Y_points[:,1]
train_total    = train_pred_pts.sum(axis=1)
true_total     = Y_points.sum(axis=1)

res_margin = (true_margin - train_margin)
res_total  = (true_total  - train_total)

sigma_margin = max(5.0, float(np.nanstd(res_margin, ddof=1)) * 0.9)
sigma_total  = max(8.0, float(np.nanstd(res_total,  ddof=1)) * 0.9)

# Out-of-fold (OOF) CV snapshots for Win/Cover/Over (derived from Points/Line/OU)
def oof_ap_snapshots(df_src: pd.DataFrame, folds: int = 5):
    # Build y's (derived): win, cover, over
    y_win   = (pd.to_numeric(df_src[d_pts], errors="coerce").values >
               pd.to_numeric(df_src[d_opp_pts], errors="coerce").values).astype(int)
    # Optional columns may be missing in some rows, so guard with NaN -> drop at scoring time
    y_sm    = pd.to_numeric(df_src.get("SM", np.nan), errors="coerce")
    y_tot   = pd.to_numeric(df_src.get("Total Points", np.nan), errors="coerce")
    y_line  = pd.to_numeric(df_src.get(d_line, np.nan), errors="coerce") if d_line else np.nan
    y_ou    = pd.to_numeric(df_src.get(d_ou, np.nan),   errors="coerce") if d_ou   else np.nan

    kf = KFold(n_splits=min(folds, max(2, len(df_src)//5)), shuffle=True, random_state=42)
    oof_win, oof_cov, oof_ov = [], [], []
    lab_win, lab_cov, lab_ov = [], [], []

    X_all = preproc.transform(df_src)

    for tr, te in kf.split(X_all):
        m = MultiOutputRegressor(RandomForestRegressor(n_estimators=600, random_state=42, n_jobs=-1))
        m.fit(X_all[tr], Y_points[tr])
        pred = m.predict(X_all[te])
        pmargin = pred[:,0] - pred[:,1]
        ptotal  = pred.sum(axis=1)

        # spreads for fold (from train residuals)
        tr_pred = m.predict(X_all[tr])
        rm = (Y_points[tr][:,0] - Y_points[tr][:,1]) - (tr_pred[:,0] - tr_pred[:,1])
        rt = (Y_points[tr].sum(axis=1)) - tr_pred.sum(axis=1)
        sm = max(5.0, float(np.nanstd(rm, ddof=1)) * 0.9)
        stt= max(8.0, float(np.nanstd(rt, ddof=1)) * 0.9)

        # Win probs
        oof_win.extend(np.clip(normal_cdf(pmargin / sm), 0.005, 0.995))
        lab_win.extend(y_win[te])

        # Cover probs only where Line exists
        if d_line and d_line in df_src.columns:
            ln = y_line.iloc[te].values
            mask = np.isfinite(ln)
            oof_cov.extend(np.clip(normal_cdf((pmargin[mask] - ln[mask]) / sm), 0.005, 0.995))
            # derive true cover from actual points if available; else from SM label if provided
            # true cover: (Points - OppPoints + Line) > 0  => SM + Line > 0
            true_cover = ((pd.to_numeric(df_src.iloc[te][d_pts], errors="coerce").values -
                           pd.to_numeric(df_src.iloc[te][d_opp_pts], errors="coerce").values + ln) > 0).astype(int)
            lab_cov.extend(true_cover[mask])

        # Over probs only where OU exists
        if d_ou and d_ou in df_src.columns:
            ou = y_ou.iloc[te].values
            mask = np.isfinite(ou)
            oof_ov.extend(np.clip(normal_cdf((ptotal[mask] - ou[mask]) / stt), 0.005, 0.995))
            true_over = ((pd.to_numeric(df_src.iloc[te][d_pts], errors="coerce").values +
                          pd.to_numeric(df_src.iloc[te][d_opp_pts], errors="coerce").values) > ou).astype(int)
            lab_ov.extend(true_over[mask])

    def ap_safe(y_true, y_score):
        if len(y_true) == 0 or len(set(y_true)) < 2: return None
        return float(average_precision_score(y_true, y_score))

    return ap_safe(lab_win, oof_win), ap_safe(lab_cov, oof_cov), ap_safe(lab_ov, oof_ov)

ap_win, ap_cov, ap_ov = oof_ap_snapshots(df_train, folds=5)

c1, c2, c3 = st.columns(3)
with c1:
    st.write(f"OOF PR-AUC (Win): {ap_win:.3f}" if ap_win is not None else "OOF PR-AUC (Win): n/a")
with c2:
    st.write(f"OOF PR-AUC (Cover): {ap_cov:.3f}" if ap_cov is not None else "OOF PR-AUC (Cover): n/a")
with c3:
    st.write(f"OOF PR-AUC (Over): {ap_ov:.3f}" if ap_ov is not None else "OOF PR-AUC (Over): n/a")

st.caption(f"Estimated σ_margin≈{sigma_margin:.2f}, σ_total≈{sigma_total:.2f}. Cover/Over/Win come from Points/OU/Line formulas.")

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
    # no per-bet cap — we will normalize to diversify
    kelly_cap = 1.0

effective_units = max(0.0, units_base - (ladder_cont_amt if ladder_cont == "Yes" else 0.0))
st.info(f"Effective units for slate (excludes ladder continuation): {effective_units:.2f}")

# =========================================================
# FEATURE BUILD FOR SCHEDULE
# =========================================================
def schedule_row_to_feature(row: pd.Series) -> np.ndarray:
    data = {}
    # numerics
    for c in daily_numeric_cols:
        data[c] = safe_num(row.get(c), np.nan)
    # categoricals (map schedule names to daily names)
    data_map = {
        d_team: team_col,
        d_opp:  opp_col,
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
    for c in zero_missing_cols:
        if c in one.columns:
            v = pd.to_numeric(one.loc[0, c], errors="coerce")
            if v == 0: one.loc[0, c] = np.nan
    X = preproc.transform(one)
    return X

# =========================================================
# SCORE SCHEDULED GAMES
# =========================================================
rows = []
for _, r in schedule_df.iterrows():
    t = str(r[team_col]).strip()
    o = str(r[opp_col]).strip()
    X = schedule_row_to_feature(r)

    pts, oppp = rf_points.predict(X)[0]
    jitter = uniq_jitter(f"{t}|{o}|{r['__Date']}")
    pts  = max(0.0, float(pts)  + jitter)
    oppp = max(0.0, float(oppp) - jitter)

    margin = pts - oppp
    total  = pts + oppp

    # Market values
    ln  = safe_num(r.get(line_col), np.nan) if line_col else np.nan
    oul = safe_num(r.get(ou_col),   np.nan) if ou_col   else np.nan
    ml  = safe_num(r.get(ml_col),   np.nan) if ml_col   else np.nan

    # Derived probabilities
    win_p   = np.clip(normal_cdf(margin / sigma_margin), 0.005, 0.995)
    cover_p = np.clip(normal_cdf((margin - ln) / sigma_margin), 0.005, 0.995) if np.isfinite(ln)  else np.nan
    over_p  = np.clip(normal_cdf((total  - oul)/ sigma_total ), 0.005, 0.995) if np.isfinite(oul) else np.nan

    ml_dec = american_to_decimal(ml) if np.isfinite(ml) else np.nan
    ml_ev  = (win_p * (ml_dec - 1.0) - (1.0 - win_p)) if np.isfinite(ml_dec) else np.nan

    spread_dec = 1.909 if np.isfinite(ln) else np.nan
    spread_ev  = (cover_p * (spread_dec - 1.0) - (1.0 - cover_p)) if np.isfinite(spread_dec) else np.nan

    ou_dec = 1.909 if np.isfinite(oul) else np.nan
    if np.isfinite(ou_dec) and np.isfinite(over_p):
        over_ev  = (over_p * (ou_dec - 1.0) - (1.0 - over_p))
        under_p  = 1.0 - over_p
        under_ev = (under_p * (ou_dec - 1.0) - (1.0 - under_p))
        if over_ev >= under_ev:
            ou_pick, ou_prob, ou_ev, ou_side_val = "Over", over_p, over_ev, oul
        else:
            ou_pick, ou_prob, ou_ev, ou_side_val = "Under", under_p, under_ev, oul
    else:
        ou_pick, ou_prob, ou_ev, ou_side_val = None, np.nan, np.nan, None

    # choose best edge by EV
    candidates = []
    if np.isfinite(spread_ev): candidates.append(("Spread", spread_ev, cover_p, spread_dec, ln))
    if np.isfinite(ml_ev):     candidates.append(("Moneyline", ml_ev, win_p, ml_dec, ml))
    if np.isfinite(ou_ev):     candidates.append((ou_pick or "OU", ou_ev, ou_prob, ou_dec, ou_side_val))
    best = max(candidates, key=lambda z: z[1], default=None)

    rows.append({
        "Date": r["__Date"],
        "Team": t, "Opponent": o,
        "Pred_Points": round(pts, 1), "Pred_Opp_Points": round(oppp, 1),
        "Pred_Margin": round(margin, 1), "Pred_Total": round(total, 1),
        "Line": ln if np.isfinite(ln) else None,
        "OU_Line": oul if np.isfinite(oul) else None,
        "ML": ml if np.isfinite(ml) else None,
        "WinProb": win_p, "CoverProb": cover_p, "OverProb": over_p,
        "BestMarket": best[0] if best else None,
        "BestEV": best[1] if best else np.nan,
        "BestProb": best[2] if best else np.nan,
        "BestDecOdds": best[3] if best else np.nan,
        "BestLineVal": best[4] if best else None,
        "RowObj": r
    })

pred_df = pd.DataFrame(rows)
if pred_df.empty:
    st.info("No priced games available.")
    st.stop()

# Priority flags for sorting
def is_true(v):
    s = str(v).strip().upper()
    return s not in ("0","","N","FALSE","NAN")

pred_df["IsTop25"] = schedule_df[top25_col].apply(is_true).values if top25_col in schedule_df.columns else False
pred_df["IsMM"]    = schedule_df[mm_col].apply(is_true).values if mm_col in schedule_df.columns else False
conf_vals = schedule_df[opp_conf_col] if opp_conf_col in schedule_df.columns else schedule_df.get(conf_col, pd.Series([""]))
pred_df["IsP5"] = conf_vals.astype(str).str.upper().isin(["SEC","BIG TEN","B1G","BIG 12","ACC","PAC-12","PAC 12"]).values

pred_df["_prio"] = pred_df.apply(lambda r: (0 if r["IsTop25"] else 1,
                                            0 if r["IsMM"] else 1,
                                            0 if r["IsP5"] else 1,
                                            r["Date"]), axis=1)
pred_df = pred_df.sort_values(by=["_prio","BestEV"], ascending=[True, False]).reset_index(drop=True)

# =========================================================
# SLATE & DIVERSIFIED STAKING (no per-bet cap; normalize by EV)
# =========================================================
slate = pred_df.dropna(subset=["BestEV"]).copy()
slate = slate[slate["BestEV"] > 0].copy()
slate = slate.sort_values(by=["_prio","BestEV"], ascending=[True, False]).reset_index(drop=True)

N = min(max_bets, max(min_bets, len(slate)))
slate = slate.head(N).copy()

# raw weights from EV * probability (smoother than Kelly)
raw_weights = slate["BestEV"].clip(lower=0) * slate["BestProb"].clip(lower=0.01, upper=0.99)
w_sum = raw_weights.sum()
if w_sum <= 0:
    # flat fallback: 15% of units split across
    slate["Stake"] = round((0.15 * effective_units) / max(1, len(slate)), 2)
else:
    slate["Stake"] = (raw_weights / w_sum) * effective_units
    slate["Stake"] = slate["Stake"].round(2)

# Remainder (rounding)
spent = float(slate["Stake"].sum())
remainder_units = round(max(0.0, effective_units - spent), 2)
if remainder_units > 0 and len(slate) > 0:
    slate.loc[slate.index[0], "Stake"] += remainder_units
    slate["Stake"] = slate["Stake"].round(2)

# =========================================================
# PRESENTATION — Suggested Slate (clean table + remainder rule)
# =========================================================
st.markdown("---")
st.markdown("### Suggested Slate (Model-Driven)")

def pick_label(r):
    if r["BestMarket"] in ("Over","Under","OU"):
        side = r["BestMarket"]
        if side == "OU":
            side = "Over" if (r.get("OverProb", np.nan) >= 0.5) else "Under"
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
    plan = pd.DataFrame([{
        "GAME": f"{r['Team']} vs {r['Opponent']}",
        "BET TYPE": pick_label(r),
        "BET AMOUNT": f"{r['Stake']:.2f} UNITS"
    } for _, r in slate.iterrows()], columns=["GAME","BET TYPE","BET AMOUNT"])

    # If stake sum ≠ effective_units (tiny rounding), top ML gets remainder and note
    sum_units = float(np.sum([float(x.split()[0]) for x in plan["BET AMOUNT"]]))
    tail = round(effective_units - sum_units, 2)
    if tail > 0.0:
        ml_cand = pred_df[pred_df["BestMarket"]=="Moneyline"].copy().sort_values(["BestProb","BestEV"], ascending=[False, False])
        if not ml_cand.empty:
            top = ml_cand.iloc[0]
            plan = pd.concat([plan, pd.DataFrame([{
                "GAME": f"{top['Team']} vs {top['Opponent']}",
                "BET TYPE": "Most probable ML (remainder)",
                "BET AMOUNT": f"{tail:.2f} UNITS"
            }])], ignore_index=True)
            st.info("Remainder assigned to the most probable ML — end of smart bets.")

    st.dataframe(plan, use_container_width=True)

# =========================================================
# PARLAY IDEAS (ABOVE DRILLDOWN) — compact matrix + details dropdown
# =========================================================
st.markdown("---")
st.markdown("### Parlay Ideas")

# Source: top by BestProb then EV
parlay_source = pred_df.sort_values(["BestProb","BestEV"], ascending=[False, False]).head(20).copy()

# map aggressiveness 1..10 → legs 2..5
min_legs = int(np.interp(homerun, [1, 10], [2, 5]))
max_legs = min_legs + 1

def build_parlays(df: pd.DataFrame, min_legs: int, max_legs: int, max_sets: int = 3):
    out = []
    picks = df.head(12).to_dict("records")
    used = set()
    for L in range(min_legs, max_legs + 1):
        if len(out) >= max_sets: break
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
            acc_prob *= float(p["BestProb"])
            acc_dec  *= float(dec)
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
        # Compact matrix display
        row1 = { "Team 1": legs[0]["Team"], "Team 2": legs[1]["Team"] if len(legs) > 1 else legs[0]["Opponent"] }
        row2 = { "Team 1": int(round(legs[0]["Pred_Points"])), "Team 2": int(round(legs[1]["Pred_Points"])) if len(legs) > 1 else int(round(legs[0]["Pred_Opp_Points"])) }
        st.table(pd.DataFrame([row1, row2]))

        with st.expander(f"Parlay {i}: {len(legs)} legs — price {dec_all:.2f} (~{decimal_to_american(dec_all)}) — hit prob ~{p_all:.2f}", expanded=False):
            for leg in legs:
                st.write(f"• {leg['Team']} vs {leg['Opponent']} — {leg['BestMarket']} @ {fmt_price(leg['BestDecOdds'])} (p={leg['BestProb']:.2f}) "
                         f"— Pred {int(round(leg['Pred_Points']))}-{int(round(leg['Pred_Opp_Points']))}")

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

        left_top50 = extract_top50_from_row(rowobj)
        cL2, cR2 = st.columns(2)
        with cL2:
            st.write(f"**{r['Team']}**")
            if not left_top50:
                st.write("None in top 50")
            else:
                st.dataframe(pd.DataFrame([{"Category": k, "Rank": v} for k, v in left_top50]),
                             use_container_width=True)

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
# EXPORT SLATE
# =========================================================
st.markdown("---")
st.markdown("### Export Slate")
export_cols = ["Date","Team","Opponent","BestMarket","BestLineVal","BestDecOdds","BestProb","BestEV","Stake"]
if not slate.empty:
    export_df = slate[export_cols].copy()
    export_df["Date"] = export_df["Date"].dt.strftime("%Y-%m-%d")
    export_df = export_df.rename(columns={
        "BestLineVal": "MarketVal",
        "BestDecOdds": "DecOdds",
        "BestProb": "BetProb",
        "BestEV": "EV"
    })
    csv = export_df.to_csv(index=False)
    st.download_button("Download Betting Slate CSV", data=csv, file_name="betting_slate.csv", mime="text/csv")
else:
    st.write("No slate to export.")
