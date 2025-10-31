# APP/Pages/9_Betting.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OrdinalEncoder

# ------------------------------------------
# CONFIG
# ------------------------------------------
BASE = "Data/26_March_Madness_Databook"
SCHEDULE_FILE = os.path.join(BASE, "2026 Schedule Transfer-Table 1.csv")
DAILY_FILE = os.path.join(BASE, "Daily_predictor_data-Table 1.csv")

st.set_page_config(page_title="Betting", layout="wide")
st.title("Betting")

# ------------------------------------------
# HELPERS
# ------------------------------------------
def load_csv(path):
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

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def parse_mdy(series_like):
    return pd.to_datetime(series_like.astype(str).str.strip(), errors="coerce", format="%m/%d/%Y")

def interpret_han(v):
    if pd.isna(v): return None
    s = str(v).strip().upper()
    if s in ("H","HOME"): return "Home"
    if s in ("A","AWAY"): return "Away"
    if s in ("N","NEUTRAL") or "NEUTRAL" in s: return "Neutral"
    return None

def american_to_prob(odds):
    try: o = float(odds)
    except: return np.nan
    if o > 0:  return 100.0/(o+100.0)
    if o < 0:  return -o/(-o+100.0)
    return np.nan

def american_to_decimal(odds):
    try: o = float(odds)
    except: return np.nan
    if o > 0:  return 1.0 + (o/100.0)
    if o < 0:  return 1.0 + (100.0/(-o))
    return np.nan

def kelly_fraction(p, dec_odds, cap=0.05):
    if not np.isfinite(p) or not np.isfinite(dec_odds) or dec_odds <= 1.0:
        return 0.0
    p = max(0.0, min(1.0, float(p)))
    b = dec_odds - 1.0
    f = (p*(b+1) - 1)/b
    return float(max(0.0, min(cap, f)))

def safe_num(x, default=0.0):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except:
        return default

def clean_rank_value(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else 51.0
    except:
        return 51.0

def extract_top50(row):
    out = []
    for c in row.index:
        cu = c.upper()
        if ("RANK" in cu) or cu.endswith(" RANK") or cu.endswith("_RANK"):
            rv = clean_rank_value(row.get(c))
            if rv <= 50:
                out.append((c, int(rv)))
    return sorted(out, key=lambda kv: kv[1])

# ------------------------------------------
# LOAD DATA
# ------------------------------------------
schedule_df = load_csv(SCHEDULE_FILE)
daily_df = load_csv(DAILY_FILE)
if schedule_df is None or daily_df is None:
    st.stop()

# ------------------------------------------
# COLUMN DETECTION (Schedule/Daily)
# ------------------------------------------
team_col = find_col(schedule_df, ["Team","Teams"])
opp_col  = find_col(schedule_df, ["Opponent","Opp","opponent"])
date_col = find_col(schedule_df, ["Date"])
han_col  = find_col(schedule_df, ["HAN","Home/Away","HomeAway","Loc","Location"])
conf_col = find_col(schedule_df, ["Conference","Conf"])
opp_conf_col = find_col(schedule_df, ["Opponent Conference","Opp Conference","Opp Conf"])
coach_col = find_col(schedule_df, ["Coach Name","Coach","Coach_Name"])
opp_coach_col = find_col(schedule_df, ["Opponent Coach","Opp Coach"])

line_col = find_col(schedule_df, ["Line"])
ml_col   = find_col(schedule_df, ["ML","Moneyline","Money Line"])
ou_col   = find_col(schedule_df, ["Over/Under Line","OverUnder","Over Under Line","O/U"])

top25_col = find_col(schedule_df, ["Top 25 Opponent","Top25","Top 25"])
mm_col    = find_col(schedule_df, ["March Madness Opponent","March Madness"])

if team_col is None or opp_col is None or date_col is None:
    st.error("Schedule must include Team, Opponent, and Date.")
    st.stop()

# Parse dates strictly as MM/DD/YYYY
schedule_df["__Date"] = parse_mdy(schedule_df[date_col])
schedule_df = schedule_df.dropna(subset=["__Date"]).copy()

# De-duplicate mirrored matchups on same date — prefer the row with HAN=Home if duplication
keep = []
seen = {}
for idx, r in schedule_df.iterrows():
    t = str(r[team_col]).strip()
    o = str(r[opp_col]).strip()
    d = r["__Date"].date()
    key = tuple(sorted([t.lower(), o.lower()])) + (d,)
    han = interpret_han(r.get(han_col)) if han_col in r.index else None
    if key not in seen:
        seen[key] = (idx, han == "Home")
    else:
        prev_idx, prev_is_home = seen[key]
        cur_is_home = (han == "Home")
        # replace previous if current is Home and previous wasn't
        if cur_is_home and not prev_is_home:
            seen[key] = (idx, True)
# finalize rows to keep
keep = [pair[0] for pair in seen.values()]
schedule_df = schedule_df.loc[keep].reset_index(drop=True)

# ------------------------------------------
# TRAINING SET FROM DAILY (Single multi-output model + win classifier)
# ------------------------------------------
d_team_col = find_col(daily_df, ["Team","Teams","team"])
d_opp_col  = find_col(daily_df, ["Opponent","Opp","opponent"])
d_pts_col  = find_col(daily_df, ["Points"])
d_opp_pts_col = find_col(daily_df, ["Opp Points","Opp_Points","OppPoints"])
if d_team_col is None or d_opp_col is None or d_pts_col is None or d_opp_pts_col is None:
    st.error("Daily predictor must have Team, Opponent, Points, Opp Points.")
    st.stop()

# Features: ALL numeric daily cols except targets
daily_numeric = daily_df.select_dtypes(include=[np.number]).copy()
numeric_feats = [c for c in daily_numeric.columns if c not in (d_pts_col, d_opp_pts_col)]

# Categoricals: team/opponent/coach/conferences (if present)
cat_cols = []
for c in [d_team_col, d_opp_col, "Coach Name","Coach","Conference","Opponent Conference","Opp Conference","Opponent Coach"]:
    if c and c in daily_df.columns:
        cat_cols.append(c)
cat_cols = list(dict.fromkeys(cat_cols))

X_num_rows, X_cat_rows, Y_points_rows, Y_win_rows = [], [], [], []
for _, r in daily_df.iterrows():
    t = r.get(d_team_col); o = r.get(d_opp_col)
    if pd.isna(t) or pd.isna(o): continue
    pts = safe_num(r.get(d_pts_col), np.nan)
    opp = safe_num(r.get(d_opp_pts_col), np.nan)
    if not np.isfinite(pts) or not np.isfinite(opp): continue

    nums = pd.to_numeric(r[numeric_feats], errors="coerce").fillna(0.0).values
    cats = [str(r.get(c)) if c in r.index else "" for c in cat_cols]
    X_num_rows.append(nums)
    X_cat_rows.append(cats)
    Y_points_rows.append([pts, opp])
    Y_win_rows.append(1 if pts > opp else 0)

if len(X_num_rows) == 0:
    st.error("No valid rows in daily predictor after cleaning.")
    st.stop()

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_cat = enc.fit_transform(np.array(X_cat_rows, dtype=object))
X_num = np.vstack(X_num_rows)
X_train = np.hstack([X_num, X_cat])
Y_points = np.array(Y_points_rows, dtype=float)
Y_win = np.array(Y_win_rows, dtype=int)

# Models
rf_multi = MultiOutputRegressor(RandomForestRegressor(n_estimators=600, random_state=42))
rf_multi.fit(X_train, Y_points)

clf = RandomForestClassifier(n_estimators=500, random_state=42)
clf.fit(X_train, Y_win)

# Residual sigmas for Gaussian probability model
train_pred = rf_multi.predict(X_train)
train_margin = train_pred[:,0] - train_pred[:,1]
true_margin  = Y_points[:,0] - Y_points[:,1]
sigma_m = float(np.std(true_margin - train_margin)) if len(true_margin) > 3 else 10.0

train_total = train_pred[:,0] + train_pred[:,1]
true_total  = Y_points[:,0] + Y_points[:,1]
sigma_t = float(np.std(true_total - train_total)) if len(true_total) > 3 else 16.0

st.success(f"Models trained on {X_train.shape[0]} games | σ_margin≈{sigma_m:.2f}, σ_total≈{sigma_t:.2f}")

# ------------------------------------------
# USER CONTROLS
# ------------------------------------------
st.markdown("### Bankroll & Strategy")
cA, cB, cC = st.columns(3)
with cA:
    units_base = st.number_input("Units of betting", min_value=1.0, value=100.0, step=1.0)
    bad_day_floor = st.number_input("Units floor (95% target)", min_value=0.0, value=65.5, step=0.5)
with cB:
    consistent_target = st.number_input("Units target (70% day)", min_value=0.0, value=140.0, step=1.0)
    homerun = st.slider("Homerun hitter (parlay aggressiveness)", 1, 10, 7)
with cC:
    ladder_start = st.selectbox("Ladder Starter", ["No","Yes"], index=1)
    ladder_cont  = st.selectbox("Ladder Continuation", ["No","Yes"], index=0)
    ladder_amt   = st.number_input("Ladder Continuation Units (excluded from base)", min_value=0.0, value=0.0, step=1.0)

cD, cE = st.columns(2)
with cD:
    min_bets, max_bets = st.slider("Bets minimum / maximum", 4, 100, (10, 25))
with cE:
    kelly_cap = st.slider("Max stake per bet (Kelly cap, % of units)", 1, 10, 5) / 100.0

effective_units = max(0.0, units_base - (ladder_amt if ladder_cont == "Yes" else 0.0))
st.info(f"Effective units for slate (excludes ladder continuation): {effective_units:.2f}")

# ------------------------------------------
# FEATURE BUILDER FOR SCHEDULE ROWS (match training schema)
# ------------------------------------------
sched_numeric = schedule_df.select_dtypes(include=[np.number]).fillna(0.0)
def row_to_features(row):
    # numeric in the order of numeric_feats
    num_vec = np.array([safe_num(row.get(c), 0.0) for c in numeric_feats], dtype=float)
    # categorical in the order of cat_cols
    cat_vals = []
    for c in cat_cols:
        if c in row.index:
            cat_vals.append(str(row.get(c)))
        elif c == d_team_col:
            cat_vals.append(str(row.get(team_col)))
        elif c == d_opp_col:
            cat_vals.append(str(row.get(opp_col)))
        elif ("Opponent" in c) and (opp_col in row.index):
            cat_vals.append(str(row.get(opp_col)))
        else:
            # Map common schedule variants
            if c in ("Coach Name","Coach","Coach_Name") and coach_col in row.index:
                cat_vals.append(str(row.get(coach_col)))
            elif c in ("Opponent Coach","Opp Coach") and opp_coach_col in row.index:
                cat_vals.append(str(row.get(opp_coach_col)))
            elif c in ("Conference","Conf") and conf_col in row.index:
                cat_vals.append(str(row.get(conf_col)))
            elif c in ("Opponent Conference","Opp Conference") and opp_conf_col in row.index:
                cat_vals.append(str(row.get(opp_conf_col)))
            else:
                cat_vals.append("")
    cat_arr = enc.transform([cat_vals])
    x = np.hstack([num_vec, cat_arr.ravel()])
    # Align with training width
    if x.shape[0] != X_train.shape[1]:
        tmp = np.zeros((X_train.shape[1],), dtype=float)
        m = min(len(tmp), len(x))
        tmp[:m] = x[:m]
        x = tmp
    return x

# ------------------------------------------
# PREDICT, PRICE, AND BUILD SLATE
# ------------------------------------------
from math import erf, sqrt
def Phi(z):  # standard normal CDF
    return 0.5*(1.0 + erf(z/sqrt(2.0)))

rows = []
for _, r in schedule_df.iterrows():
    t = str(r[team_col]).strip()
    o = str(r[opp_col]).strip()
    date_dt = r["__Date"]

    x = row_to_features(r)
    pts_for, pts_against = rf_multi.predict(x.reshape(1,-1))[0]
    pts_for = max(0.0, float(pts_for))
    pts_against = max(0.0, float(pts_against))
    margin = pts_for - pts_against
    total  = pts_for + pts_against

    # Win prob from classifier
    win_prob = float(clf.predict_proba(x.reshape(1,-1))[0][1])

    # Market lines (from team perspective, e.g., BYU -7.5 means team is favored by 7.5)
    line = safe_num(r.get(line_col), np.nan) if line_col else np.nan
    ml   = safe_num(r.get(ml_col),   np.nan) if ml_col   else np.nan
    ou   = safe_num(r.get(ou_col),   np.nan) if ou_col   else np.nan

    # Spread probability via Gaussian residual
    cover_prob = np.nan
    if np.isfinite(line):
        z = (margin - line) / max(1e-6, sigma_m)
        cover_prob = Phi(z)  # team - line >= 0 -> covers
    # Moneyline price/implied
    ml_dec = american_to_decimal(ml) if np.isfinite(ml) else np.nan
    # Totals probability
    ou_pick, ou_prob = None, np.nan
    if np.isfinite(ou):
        zt = (total - ou) / max(1e-6, sigma_t)
        prob_over = Phi(zt)
        if total >= ou:
            ou_pick, ou_prob = "Over", prob_over
        else:
            ou_pick, ou_prob = "Under", 1.0 - prob_over

    # Edges & Kelly (spreads/totals assumed -110 if no price; ML uses market price)
    spread_dec = 1.909 if np.isfinite(line) else np.nan
    spread_edge = (cover_prob - 1.0/spread_dec) if np.isfinite(cover_prob) else np.nan
    spread_kelly = kelly_fraction(cover_prob, spread_dec, cap=kelly_cap) if np.isfinite(cover_prob) else 0.0

    ml_edge = (win_prob - 1.0/ml_dec) if np.isfinite(ml_dec) else np.nan
    ml_kelly = kelly_fraction(win_prob, ml_dec, cap=kelly_cap) if np.isfinite(ml_dec) else 0.0

    ou_dec = 1.909 if np.isfinite(ou_prob) else np.nan
    ou_edge = (ou_prob - 1.0/ou_dec) if np.isfinite(ou_prob) else np.nan
    ou_kelly = kelly_fraction(ou_prob, ou_dec, cap=kelly_cap) if np.isfinite(ou_prob) else 0.0

    # Priority: Top25 -> MM -> P5
    is_top25 = False
    if top25_col and top25_col in schedule_df.columns:
        v = r.get(top25_col)
        is_top25 = (str(v).strip() not in ("0","","N","FALSE","NaN"))
    is_mm = False
    if mm_col and mm_col in schedule_df.columns:
        v = r.get(mm_col)
        is_mm = (str(v).strip() not in ("0","","N","FALSE","NaN"))
    conf = r.get(opp_conf_col) if opp_conf_col in r.index else r.get(conf_col)
    cu = str(conf).strip().upper() if conf is not None else ""
    is_p5 = cu in ("SEC","BIG TEN","B1G","BIG 12","ACC")

    # pick best lane by highest positive edge
    choices = []
    if np.isfinite(spread_edge): choices.append(("Spread", spread_edge, cover_prob, spread_dec, spread_kelly, line))
    if np.isfinite(ml_edge):     choices.append(("Moneyline", ml_edge, win_prob, ml_dec, ml_kelly, ml))
    if np.isfinite(ou_edge):     choices.append((ou_pick or "OU", ou_edge, ou_prob, ou_dec, ou_kelly, ou))
    best = max(choices, key=lambda z: (z[1] if np.isfinite(z[1]) else -1e9), default=None)

    rows.append({
        "Date": date_dt,
        "Team": t,
        "Opponent": o,
        "Pred_Points": round(pts_for,1),
        "Pred_Opp_Points": round(pts_against,1),
        "Pred_Margin": round(margin,1),
        "Pred_Total": round(total,1),
        "Win_Prob": round(win_prob,3),
        "Line": line if np.isfinite(line) else None,
        "Moneyline": int(ml) if np.isfinite(ml) else None,
        "OU_Line": ou if np.isfinite(ou) else None,
        "Cover_Prob": cover_prob,
        "OU_Pick": ou_pick,
        "OU_Prob": ou_prob,
        "Spread_Edge": spread_edge,
        "ML_Edge": ml_edge,
        "OU_Edge": ou_edge,
        "Spread_Kelly": spread_kelly,
        "ML_Kelly": ml_kelly,
        "OU_Kelly": ou_kelly,
        "IsTop25": is_top25,
        "IsMM": is_mm,
        "IsP5": is_p5,
        "BestMarket": best[0] if best else None,
        "BestEdge": best[1] if best else np.nan,
        "BestProb": best[2] if best else np.nan,
        "BestDecOdds": best[3] if best else np.nan,
        "BestLineVal": best[5] if best else None,
        "RowObj": r
    })

pred_df = pd.DataFrame(rows)
if pred_df.empty:
    st.info("No games available.")
    st.stop()

# ------------------------------------------
# SLATE SELECTION & SIZING
# ------------------------------------------
pred_df["_prio"] = pred_df.apply(lambda r: (
    0 if r["IsTop25"] else 1,
    0 if r["IsMM"] else 1,
    0 if r["IsP5"] else 1,
    r["Date"]
), axis=1)

slate = pred_df.dropna(subset=["BestEdge"]).copy()
slate = slate[slate["BestEdge"] > 0].sort_values(["_prio","BestEdge"], ascending=[True, False]).reset_index(drop=True)

N = min(max_bets, max(min_bets, len(slate)))
slate = slate.head(N).copy()

def stake_units(row):
    f = float(row["BestKelly"])
    if row["BestMarket"] == "Spread": f = row["Spread_Kelly"]
    elif row["BestMarket"] == "Moneyline": f = row["ML_Kelly"]
    else: f = row["OU_Kelly"]
    if not np.isfinite(f) or f <= 0: return 0.0
    return round(f * effective_units, 2)

slate["Stake"] = slate.apply(stake_units, axis=1)
if slate["Stake"].sum() == 0 and len(slate) > 0:
    # flat 15% allocation across slate if Kelly couldn't size (e.g., missing prices)
    target = 0.15 * effective_units
    slate["Stake"] = round(target/len(slate), 2)

# ------------------------------------------
# LADDER PICKS (best high-prob bet with price in [-150, +150])
# ------------------------------------------
def pick_ladder(df, exclude=None):
    candidates = []
    for _, r in df.iterrows():
        market = r["BestMarket"]
        if market is None: continue
        key = (r["Team"], r["Opponent"], r["Date"], market)
        if exclude and key == exclude: continue
        dec = r["BestDecOdds"]
        if not np.isfinite(dec):  # no price → use -110 equivalent if spread/OU
            if market in ("Spread","Over","Under","OU"):
                dec = 1.909
        if not np.isfinite(dec): continue
        # map dec to American to test range; odd conversion: dec>=2 → positive
        amer = (dec-1)*100 if dec>=2 else -100/(dec-1)
        if -150 <= amer <= 150:
            candidates.append((r["BestProb"], -abs(amer), r))
    if not candidates: return None
    candidates.sort(key=lambda z: (z[0], z[1]), reverse=True)
    return candidates[0][2]

ladder_starter = pick_ladder(pred_df) if ladder_start == "Yes" else None
exclude_key = (ladder_starter["Team"], ladder_starter["Opponent"], ladder_starter["Date"], ladder_starter["BestMarket"]) if ladder_starter is not None else None
ladder_cont_row = pick_ladder(pred_df, exclude=exclude_key) if ladder_cont == "Yes" else None

# ------------------------------------------
# DISPLAY SLATE
# ------------------------------------------
st.markdown("### Suggested Slate")
def fmt_dec(dec):
    if not np.isfinite(dec): return ""
    if dec >= 2: return f"+{int(round((dec-1)*100))}"
    return f"{int(round(-100/(dec-1)))}"

if slate.empty:
    st.info("No +EV bets identified with current markets.")
else:
    view = slate[["Date","Team","Opponent","BestMarket","BestLineVal","Pred_Margin","Pred_Total","Win_Prob","BestProb","BestEdge","BestDecOdds","Stake"]].copy()
    view = view.rename(columns={"BestLineVal":"Market","BestDecOdds":"Price (dec)","BestProb":"Bet Prob"})
    view["Date"] = view["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(view, use_container_width=True)

st.markdown("---")
st.markdown("### Ladders")
c1, c2 = st.columns(2)
with c1:
    st.write("Ladder Starter" + (" (enabled)" if ladder_start=="Yes" else " (disabled)"))
    if ladder_starter is not None:
        r = ladder_starter
        st.write(f"{r['Date'].strftime('%Y-%m-%d')}: {r['Team']} vs {r['Opponent']} — {r['BestMarket']} @ {fmt_dec(r['BestDecOdds'])} — p={r['BestProb']:.2f}")
        st.write("Stake: choose your day-1 ladder unit (not from slate units).")
    else:
        st.write("No suitable starter identified in [-150, +150].")
with c2:
    st.write("Ladder Continuation" + (" (enabled)" if ladder_cont=="Yes" else " (disabled)"))
    if ladder_cont_row is not None:
        r = ladder_cont_row
        st.write(f"{r['Date'].strftime('%Y-%m-%d')}: {r['Team']} vs {r['Opponent']} — {r['BestMarket']} @ {fmt_dec(r['BestDecOdds'])} — p={r['BestProb']:.2f}")
        st.write(f"Continuation Stake: {ladder_amt:.2f} units (excluded from slate units).")
    else:
        st.write("No suitable continuation identified in [-150, +150].")

# ------------------------------------------
# DRILLDOWN (Team details + Top-50 ranks split)
# ------------------------------------------
st.markdown("---")
st.markdown("### Drilldown (click a matchup for details)")

for _, r in pred_df.sort_values(["_prio","BestEdge"], ascending=[True, False]).iterrows():
    label = f"{r['Date'].strftime('%b %d, %Y')} — {r['Team']} vs {r['Opponent']} — Pred: {int(round(r['Pred_Points']))}-{int(round(r['Pred_Opp_Points']))}"
    with st.expander(label, expanded=False):
        row = r["RowObj"]
        t_conf = row.get(conf_col) if conf_col in row.index else ""
        o_conf = row.get(opp_conf_col) if opp_conf_col in row.index else ""
        t_coach = row.get(coach_col) if coach_col in row.index else ""
        o_coach = row.get(opp_coach_col) if opp_coach_col in row.index else ""
        lw, rw = st.columns(2)
        with lw:
            st.markdown(f"**{r['Team']}**")
            st.write(f"Coach: {t_coach}")
            st.write(f"Conference: {t_conf}")
            tf = extract_top50(row)
            if tf:
                st.dataframe(pd.DataFrame([{"Category": k, "Rank": v} for k,v in tf]), use_container_width=True)
            else:
                st.write("No listed top-50 ranks.")
        with rw:
            st.markdown(f"**{r['Opponent']}**")
            st.write(f"Coach: {o_coach}")
            st.write(f"Conference: {o_conf}")
            # Opponent top-50 = look for OPP_* rank fields ≤ 50
            opp_top = []
            for c in row.index:
                cu = c.upper()
                if ("OPP_" in cu) and (("RANK" in cu) or cu.endswith(" RANK") or cu.endswith("_RANK")):
                    rv = clean_rank_value(row.get(c))
                    if rv <= 50:
                        opp_top.append((c, int(rv)))
            opp_top = sorted(opp_top, key=lambda kv: kv[1])
            if opp_top:
                st.dataframe(pd.DataFrame([{"Category": k, "Rank": v} for k,v in opp_top]), use_container_width=True)
            else:
                st.write("No listed top-50 ranks.")

# ------------------------------------------
# EXPORT
# ------------------------------------------
st.markdown("---")
st.markdown("### Export Slate")
export_cols = ["Date","Team","Opponent","BestMarket","BestLineVal","BestDecOdds","Bet Prob","BestEdge","Stake"]
out = slate.rename(columns={"BestProb":"Bet Prob"})[export_cols].copy()
out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
st.download_button("Download Slate CSV", data=out.to_csv(index=False), file_name="betting_slate.csv", mime="text/csv")
