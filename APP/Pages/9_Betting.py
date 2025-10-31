# APP/Pages/7_Betting.py
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
    # Force MM/DD/YYYY (your files are in this format)
    return pd.to_datetime(
        series_like.astype(str).str.strip(),
        errors="coerce",
        format="%m/%d/%Y"
    )

def interpret_han(v):
    if pd.isna(v): return None
    s = str(v).strip().upper()
    if s in ("H","HOME"): return "Home"
    if s in ("A","AWAY"): return "Away"
    if s in ("N","NEUTRAL") or "NEUTRAL" in s: return "Neutral"
    return None

def american_to_prob(odds):
    """
    Convert American odds to implied probability.
    """
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    elif o < 0:
        return -o / (-o + 100.0)
    return np.nan

def american_to_decimal(odds):
    """
    Convert American odds to decimal odds (>1.0).
    """
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 1.0 + (o / 100.0)
    elif o < 0:
        return 1.0 + (100.0 / -o)
    return np.nan

def kelly_fraction(edge_prob, dec_odds, cap=0.05):
    """
    Kelly fraction based on model probability and decimal odds.
    Returns fraction of bankroll to stake (capped).
    If inputs are invalid or edge <= 0, returns 0.
    """
    if not np.isfinite(edge_prob) or not np.isfinite(dec_odds):
        return 0.0
    p = max(0.0, min(1.0, edge_prob))
    b = dec_odds - 1.0
    if b <= 0:
        return 0.0
    f = (p * (b + 1) - 1) / b  # classic Kelly
    if f <= 0:
        return 0.0
    return float(min(cap, f))

def safe_num(x, default=0.0):
    try:
        v = float(x)
        if np.isnan(v): return default
        return v
    except Exception:
        return default

def clean_rank_value(x):
    # treat NA/"N/A"/blank as 51+ (out of top-50 scope)
    try:
        v = float(x)
        if np.isnan(v): return 51.0
        return v
    except Exception:
        return 51.0

def extract_team_top50(row):
    """
    From a schedule row with many *_Rank columns, return a list of (Category, Rank)
    for columns where the numeric rank <= 50.
    """
    out = []
    for c in row.index:
        cu = c.upper()
        if "RANK" in cu or cu.endswith(" RANK") or cu.endswith("_RANK"):
            rank_val = clean_rank_value(row.get(c))
            if rank_val <= 50:
                out.append((c, int(rank_val)))
    # sort best to worst
    return sorted(out, key=lambda x: x[1])

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
team_col = find_col(schedule_df, ["Team", "Teams"])
opp_col  = find_col(schedule_df, ["Opponent", "Opp", "opponent"])
date_col = find_col(schedule_df, ["Date"])
han_col  = find_col(schedule_df, ["HAN","Home/Away","HomeAway","Loc","Location"])
conf_col = find_col(schedule_df, ["Conference"])
opp_conf_col = find_col(schedule_df, ["Opponent Conference", "Opp Conference"])
coach_col = find_col(schedule_df, ["Coach Name","Coach","Coach_Name"])
opp_coach_col = find_col(schedule_df, ["Opponent Coach","Opp Coach"])

line_col = find_col(schedule_df, ["Line"])
ml_col = find_col(schedule_df, ["ML","Moneyline","Money Line"])
ou_col = find_col(schedule_df, ["Over/Under Line","OverUnder","Over Under Line","O/U"])

top25_col = find_col(schedule_df, ["Top 25 Opponent","Top25","Top 25"])
mm_col    = find_col(schedule_df, ["March Madness Opponent","March Madness"])

if team_col is None or opp_col is None or date_col is None:
    st.error("Schedule must include Team, Opponent, and Date.")
    st.stop()

# Parse dates strictly as MM/DD/YYYY per your requirement
schedule_df["__Date"] = parse_mdy(schedule_df[date_col])
schedule_df = schedule_df.dropna(subset=["__Date"]).copy()

# De-duplicate mirrored matchups on same date (keep the HOME row if available)
seen = set()
keep_rows = []
for idx, r in schedule_df.iterrows():
    t = str(r[team_col]).strip()
    o = str(r[opp_col]).strip()
    d = r["__Date"].date()
    key = tuple(sorted([t.lower(), o.lower()])) + (d,)
    if key in seen:
        # Prefer the row whose HAN is Home if one of the dupes is Home; else first seen
        prev_idx = keep_rows[-1] if keep_rows else None
        # simple policy: keep the first seen; you can enhance to check HAN==Home
        continue
    seen.add(key)
    keep_rows.append(idx)
schedule_df = schedule_df.loc[keep_rows].reset_index(drop=True)

# ------------------------------------------
# TRAINING SET FROM DAILY (Targets: Points & Opp Points, plus Win)
# ------------------------------------------
# Identify team/opp columns in daily
d_team_col = find_col(daily_df, ["Team","Teams","team"])
d_opp_col  = find_col(daily_df, ["Opponent","Opp","opponent"])
d_pts_col  = find_col(daily_df, ["Points"])
d_opp_pts_col = find_col(daily_df, ["Opp Points","Opp_Points","OppPoints"])

if d_team_col is None or d_opp_col is None or d_pts_col is None or d_opp_pts_col is None:
    st.error("Daily predictor must have Team, Opponent, Points, Opp Points.")
    st.stop()

# Features: all numeric columns *except* Points/Opp Points
daily_numeric = daily_df.select_dtypes(include=[np.number]).copy()
numeric_feature_cols = [c for c in daily_numeric.columns if c not in (d_pts_col, d_opp_pts_col)]

# Categorical features to add (ordinal encode)
cat_cols = []
for c in [d_team_col, d_opp_col, "Coach Name", "Coach", "Conference", "Opponent Conference", "Opp Conference", "Opponent Coach"]:
    if c and c in daily_df.columns:
        cat_cols.append(c)
cat_cols = list(dict.fromkeys(cat_cols))  # de-dup

# Build training matrices
X_rows = []
Y_points_rows = []
Y_win = []

for _, r in daily_df.iterrows():
    t = r.get(d_team_col)
    o = r.get(d_opp_col)
    if pd.isna(t) or pd.isna(o):
        continue

    # numeric block
    nums = pd.to_numeric(r[numeric_feature_cols], errors="coerce").fillna(0.0).values

    # categorical block
    cat_vals = []
    for c in cat_cols:
        cat_vals.append(str(r.get(c)) if c in r.index else "")

    # targets
    pts = safe_num(r.get(d_pts_col), default=np.nan)
    opp = safe_num(r.get(d_opp_pts_col), default=np.nan)
    if np.isnan(pts) or np.isnan(opp):
        continue
    win = 1 if pts > opp else 0

    X_rows.append((nums, cat_vals))
    Y_points_rows.append([pts, opp])
    Y_win.append(win)

if len(X_rows) == 0:
    st.error("No valid rows in daily predictor for training.")
    st.stop()

# Fit encoder over all cat rows
cat_matrix = np.array([row[1] for row in X_rows], dtype=object)
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
cat_encoded = enc.fit_transform(cat_matrix)

X_num_mat = np.vstack([row[0] for row in X_rows])
X_train = np.hstack([X_num_mat, cat_encoded])
Y_points = np.array(Y_points_rows, dtype=float)
Y_win = np.array(Y_win, dtype=int)

# Train models
rf_multi = MultiOutputRegressor(RandomForestRegressor(n_estimators=600, random_state=42))
rf_multi.fit(X_train, Y_points)

clf = RandomForestClassifier(n_estimators=500, random_state=42)
clf.fit(X_train, Y_win)

st.success(f"Models trained on {X_train.shape[0]} games with {X_train.shape[1]} features.")

# ------------------------------------------
# BUILD FEATURES FOR SCHEDULE ROWS
# ------------------------------------------
# Compose schedule feature columns with same numeric names (best effort)
sched_numeric = schedule_df.select_dtypes(include=[np.number]).copy()
sched_numeric = sched_numeric.fillna(0.0)

common_numeric = [c for c in numeric_feature_cols if c in sched_numeric.columns]
# For missing numeric features, we will append zeros in the right positions later.

def row_to_feature_vector(row):
    # numeric slice in the order of numeric_feature_cols (pad zeros for missing)
    num_vec = []
    for c in numeric_feature_cols:
        if c in row.index:
            num_vec.append(safe_num(row[c], 0.0))
        else:
            # try some opponent mapping for common fields (best-effort)
            num_vec.append(0.0)
    num_vec = np.array(num_vec, dtype=float)

    # categorical: build in the same order of cat_cols used for training, pulling from schedule
    cat_vals = []
    for c in cat_cols:
        if c in row.index:
            cat_vals.append(str(row.get(c)))
        elif c == d_team_col and team_col in row.index:
            cat_vals.append(str(row.get(team_col)))
        elif c == d_opp_col and opp_col in row.index:
            cat_vals.append(str(row.get(opp_col)))
        elif "Opponent" in c and opp_col in row.index:
            cat_vals.append(str(row.get(opp_col)))
        else:
            # try mapped alternatives
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
    # align to training width (pad/truncate)
    if x.shape[0] != X_train.shape[1]:
        tmp = np.zeros((X_train.shape[1],), dtype=float)
        m = min(len(tmp), len(x))
        tmp[:m] = x[:m]
        x = tmp
    return x

# ------------------------------------------
# USER CONTROLS (Bankroll, Targets, Strategy)
# ------------------------------------------
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
    ladder_cont = st.selectbox("Ladder Continuation", ["No","Yes"], index=0)
    ladder_cont_amt = st.number_input("Ladder Continuation Units (excluded from base)", min_value=0.0, value=0.0, step=1.0)

colD, colE = st.columns(2)
with colD:
    min_bets, max_bets = st.slider("Bets minimum / maximum", 4, 100, (10, 25))
with colE:
    kelly_cap = st.slider("Max stake per bet (Kelly cap, % of units)", 1, 10, 5) / 100.0

effective_units = max(0.0, units_base - (ladder_cont_amt if ladder_cont == "Yes" else 0.0))
st.info(f"Effective units for slate (excludes ladder continuation): {effective_units:.2f}")

# ------------------------------------------
# PREDICTIONS + PRICING
# ------------------------------------------
rows = []
for idx, r in schedule_df.iterrows():
    t = str(r[team_col]).strip()
    o = str(r[opp_col]).strip()
    date_dt = r["__Date"]

    x = row_to_feature_vector(r)
    pred_pts, pred_opp = rf_multi.predict(x.reshape(1, -1))[0]
    pred_pts = max(0.0, float(pred_pts))
    pred_opp = max(0.0, float(pred_opp))
    margin = pred_pts - pred_opp
    total  = pred_pts + pred_opp

    # win prob from classifier
    win_prob = float(clf.predict_proba(x.reshape(1, -1))[0][1])

    # market lines (team perspective)
    line = safe_num(r.get(line_col), default=np.nan) if line_col else np.nan
    moneyline = safe_num(r.get(ml_col), default=np.nan) if ml_col else np.nan
    ou_line = safe_num(r.get(ou_col), default=np.nan) if ou_col else np.nan

    # derive EVs
    # Spread (team - line)
    spread_edge = np.nan
    if np.isfinite(line):
        # Approx cover prob using margin vs line via a simple logistic proxy based on training spread stdev
        # If you want, replace with a direct cover classifier later.
        # Use a soft proxy: P(cover) ~= sigmoid((margin - line)/sigma)
        sigma = 8.5  # conservative game-to-game margin scale; tune if desired
        cover_prob = 1.0 / (1.0 + np.exp(-(margin - line) / max(1e-6, sigma)))
        # Assume -110 standard vig if true odds absent
        dec_odds = 1.909  # -110
        f = kelly_fraction(cover_prob, dec_odds, cap=kelly_cap)
        spread_edge = cover_prob - (1.0 / dec_odds)
    else:
        cover_prob, f = np.nan, 0.0
        dec_odds = np.nan

    # Moneyline EV (use market odds if present, else synth from win_prob)
    ml_dec = american_to_decimal(moneyline) if np.isfinite(moneyline) else np.nan
    ml_implied = american_to_prob(moneyline) if np.isfinite(moneyline) else np.nan
    ml_edge = np.nan
    ml_kelly = 0.0
    if np.isfinite(ml_dec):
        ml_edge = win_prob - (1.0 / ml_dec)
        ml_kelly = kelly_fraction(win_prob, ml_dec, cap=kelly_cap)

    # Over/Under
    ou_pick = ""
    ou_edge = np.nan
    ou_kelly = 0.0
    if np.isfinite(ou_line):
        # crude variance proxy for totals; tune later as data grows
        tau = 12.0
        prob_over = 1.0 / (1.0 + np.exp(-(total - ou_line) / max(1e-6, tau)))
        # assume -110 for OU if price missing
        ou_dec = 1.909
        if total > ou_line:
            ou_pick = "Over"
            ou_edge = prob_over - (1.0 / ou_dec)
            ou_kelly = kelly_fraction(prob_over, ou_dec, cap=kelly_cap)
        else:
            ou_pick = "Under"
            prob_under = 1.0 - prob_over
            ou_edge = prob_under - (1.0 / ou_dec)
            ou_kelly = kelly_fraction(prob_under, ou_dec, cap=kelly_cap)

    # Priority buckets (Top25 > MM > P5)
    is_top25 = False
    if top25_col and top25_col in schedule_df.columns:
        v = r.get(top25_col)
        is_top25 = (str(v).strip() not in ("0","", "N","FALSE","NaN"))

    is_mm = False
    if mm_col and mm_col in schedule_df.columns:
        v = r.get(mm_col)
        is_mm = (str(v).strip() not in ("0","", "N","FALSE","NaN"))

    conf = r.get(opp_conf_col) if opp_conf_col in r.index else r.get(conf_col)
    c = str(conf).strip().upper() if conf is not None else ""
    is_p5 = c in ("SEC","BIG TEN","B1G","BIG 12","ACC")

    # choose “best” bet lane by max edge
    edges = []
    if np.isfinite(spread_edge):
        edges.append(("Spread", spread_edge, cover_prob, dec_odds, f, line))
    if np.isfinite(ml_edge):
        edges.append(("Moneyline", ml_edge, win_prob, ml_dec, ml_kelly, moneyline))
    if np.isfinite(ou_edge):
        odds = 1.909
        p = (1.0 / (1.0 + np.exp(-(total - ou_line)/12.0))) if ou_pick == "Over" else (1.0 - 1.0 / (1.0 + np.exp(-(total - ou_line)/12.0)))
        edges.append((ou_pick or "OU", ou_edge, p, odds, ou_kelly, ou_line))

    best_market = max(edges, key=lambda z: (z[1] if np.isfinite(z[1]) else -1e9), default=None)

    rows.append({
        "Date": date_dt,
        "Team": t,
        "Opponent": o,
        "Pred_Points": round(pred_pts, 1),
        "Pred_Opp_Points": round(pred_opp, 1),
        "Pred_Margin": round(margin, 1),
        "Pred_Total": round(total, 1),
        "Win_Prob": round(win_prob, 3),
        "Line": line if np.isfinite(line) else None,
        "Moneyline": int(moneyline) if np.isfinite(moneyline) else None,
        "OU_Line": ou_line if np.isfinite(ou_line) else None,
        "IsTop25": is_top25,
        "IsMM": is_mm,
        "IsP5": is_p5,
        "BestMarket": best_market[0] if best_market else None,
        "BestEdge": best_market[1] if best_market else np.nan,
        "BestProb": best_market[2] if best_market else np.nan,
        "BestDecOdds": best_market[3] if best_market else np.nan,
        "BestKelly": best_market[4] if best_market else 0.0,
        "BestLineVal": best_market[5] if best_market else None,
        "RowObj": r  # keep original row for drilldown/top-50 extraction
    })

pred_df = pd.DataFrame(rows)
if pred_df.empty:
    st.info("No games available.")
    st.stop()

# ------------------------------------------
# SORTING BY PRIORITY + EDGE
# ------------------------------------------
priority_key = pred_df.apply(
    lambda r: (
        0 if r["IsTop25"] else 1,
        0 if r["IsMM"] else 1,
        0 if r["IsP5"] else 1,
        r["Date"]
    ),
    axis=1
)
pred_df = pred_df.assign(_prio=priority_key)
pred_df = pred_df.sort_values(by=["_prio","BestEdge"], ascending=[True, False]).reset_index(drop=True)

# ------------------------------------------
# LADDER PICKS
# ------------------------------------------
def pick_ladder(df, exclude_key=None):
    # pick safest high-prob bet within odds ~ -150..+150 if possible (OU or spread ok)
    candidates = []
    for _, r in df.iterrows():
        market = r["BestMarket"]
        if market is None: continue
        # ladder shouldn't duplicate starter/continuation picks
        key = (r["Team"], r["Opponent"], r["Date"], market)
        if exclude_key and key == exclude_key:
            continue

        dec = r["BestDecOdds"]
        if not np.isfinite(dec): 
            # synth a dec odds for spread/OU if missing
            dec = 1.909
        # accept near-even odds
        if 1.67 <= dec <= 2.5:  # -150 to +150 approx
            candidates.append((r["BestProb"], -abs(dec-2.0), r))  # max prob, prefer closer to EVENS

    if not candidates:
        return None
    candidates.sort(key=lambda z: (z[0], z[1]), reverse=True)
    return candidates[0][2]  # return row

ladder_starter_pick = None
ladder_cont_pick = None
ex_key = None

if ladder_start == "Yes":
    ladder_starter_pick = pick_ladder(pred_df)
    if ladder_starter_pick is not None:
        ex_key = (ladder_starter_pick["Team"], ladder_starter_pick["Opponent"], ladder_starter_pick["Date"], ladder_starter_pick["BestMarket"])

if ladder_cont == "Yes":
    ladder_cont_pick = pick_ladder(pred_df, exclude_key=ex_key)

# ------------------------------------------
# BUILD SLATE (MIN/MAX + SIZING)
# ------------------------------------------
# Rank by BestEdge descending; pick between min_bets ~ max_bets under bankroll cap
slate = pred_df.dropna(subset=["BestEdge"]).copy()
slate = slate[slate["BestEdge"] > 0].copy()
slate = slate.sort_values(["_prio","BestEdge"], ascending=[True, False]).reset_index(drop=True)

# choose N within bounds
N = min(max_bets, max(min_bets, len(slate)))
slate = slate.head(N).copy()

# stake sizing (capped Kelly on effective_units)
def stake_for_row(r):
    f = float(r["BestKelly"])
    if not np.isfinite(f) or f <= 0:
        return 0.0
    return round(f * effective_units, 2)

slate["Stake"] = slate.apply(stake_for_row, axis=1)
# if all zero (no price), allocate flat small stakes evenly to meet a conservative spend (~15% of units)
if slate["Stake"].sum() == 0 and len(slate) > 0:
    target_spend = 0.15 * effective_units
    per = round(target_spend / len(slate), 2)
    slate["Stake"] = per

# ------------------------------------------
# PARLAY SUGGESTIONS (Homerun)
# ------------------------------------------
# Build a few parlay ideas from top-probability options
parlay_source = pred_df.sort_values(["BestProb","BestEdge"], ascending=[False, False]).head(20).copy()
parlay_count = {1:(2,3), 5:(3,4), 10:(4,5)}
# interpolate legs by slider
min_legs = int(np.interp(homerun, [1,10], [2,5]))
max_legs = min_legs + 1

def build_parlays(df, min_legs, max_legs, max_sets=3):
    out = []
    picks = df.head(10).to_dict("records")
    # simple greedy combos
    used = set()
    for L in range(min_legs, max_legs+1):
        if len(out) >= max_sets: break
        legs, acc_prob, acc_dec = [], 1.0, 1.0
        for p in picks:
            key = (p["Team"], p["Opponent"], p["Date"], p["BestMarket"])
            if key in used: continue
            if not np.isfinite(p["BestProb"]): continue
            dec = p["BestDecOdds"]
            if not np.isfinite(dec): dec = 1.909
            legs.append(p)
            acc_prob *= p["BestProb"]
            acc_dec *= dec
            used.add(key)
            if len(legs) >= L: break
        if len(legs) == L and acc_prob > 0:
            out.append((legs, acc_prob, acc_dec))
    return out

parlays = build_parlays(parlay_source, min_legs, max_legs, max_sets=3)

# ------------------------------------------
# DISPLAY
# ------------------------------------------
st.markdown("### Suggested Slate")
def fmt_odds(od):
    if not np.isfinite(od): return "-110"
    # convert decimal to American-ish for display if needed
    if od >= 2.0:
        return f"+{int(round((od-1)*100))}"
    else:
        return f"{int(round(-100/(od-1)))}"

if slate.empty:
    st.info("No +EV bets identified with current markets.")
else:
    view_cols = ["Date","Team","Opponent","BestMarket","BestLineVal","Pred_Margin","Pred_Total","Win_Prob","BestProb","BestEdge","BestDecOdds","Stake"]
    show = slate[view_cols].copy()
    show = show.rename(columns={
        "BestLineVal":"Market",
        "BestDecOdds":"Price (dec)",
        "BestProb":"Bet Prob"
    })
    show["Date"] = show["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(show, use_container_width=True)

# Ladder cards
st.markdown("---")
st.markdown("### Ladders")
lc1, lc2 = st.columns(2)
with lc1:
    st.write("Ladder Starter" + (" (enabled)" if ladder_start=="Yes" else " (disabled)"))
    if ladder_starter_pick is not None:
        r = ladder_starter_pick
        st.write(f"{r['Date'].strftime('%Y-%m-%d')}: {r['Team']} vs {r['Opponent']} — {r['BestMarket']} @ {fmt_odds(r['BestDecOdds'])} — p={r['BestProb']:.2f}")
    else:
        st.write("No suitable starter identified.")

with lc2:
    st.write("Ladder Continuation" + (" (enabled)" if ladder_cont=="Yes" else " (disabled)"))
    if ladder_cont_pick is not None:
        r = ladder_cont_pick
        st.write(f"{r['Date'].strftime('%Y-%m-%d')}: {r['Team']} vs {r['Opponent']} — {r['BestMarket']} @ {fmt_odds(r['BestDecOdds'])} — p={r['BestProb']:.2f}")
        st.write(f"Continuation Stake: {ladder_cont_amt:.2f} units (excluded from slate units)")
    else:
        st.write("No suitable continuation identified.")

# Parlays
st.markdown("---")
st.markdown("### Parlay Ideas")
if not parlays:
    st.write("No parlay suggestions currently.")
else:
    for i, (legs, p_all, dec_all) in enumerate(parlays, 1):
        st.write(f"Parlay {i}: {len(legs)} legs — price {dec_all:.2f} (~{fmt_odds(dec_all)}) — hit prob ~{p_all:.2f}")
        for leg in legs:
            st.write(f"• {leg['Team']} vs {leg['Opponent']} — {leg['BestMarket']} @ {fmt_odds(leg['BestDecOdds'])} (p={leg['BestProb']:.2f})")

# ------------------------------------------
# DRILLDOWN EXPANDERS (Top-50 categories split per team)
# ------------------------------------------
st.markdown("---")
st.markdown("### Drilldown (click a matchup for team details and Top-50 categories)")

for _, r in pred_df.iterrows():
    label = f"{r['Date'].strftime('%b %d, %Y')} — {r['Team']} vs {r['Opponent']} — Pred: {int(round(r['Pred_Points']))}-{int(round(r['Pred_Opp_Points']))}"
    with st.expander(label, expanded=False):
        rowobj = r["RowObj"]
        # Meta
        t_conf = rowobj.get(conf_col) if conf_col in rowobj.index else ""
        o_conf = rowobj.get(opp_conf_col) if opp_conf_col in rowobj.index else ""
        t_coach = rowobj.get(coach_col) if coach_col in rowobj.index else ""
        o_coach = rowobj.get(opp_coach_col) if opp_coach_col in rowobj.index else ""
        # Current record if present
        t_wins = rowobj.get("Wins") if "Wins" in rowobj.index else ""
        t_losses = rowobj.get("Losses") if "Losses" in rowobj.index else ""
        o_wins = rowobj.get("Opp Wins") if "Opp Wins" in rowobj.index else ""
        o_losses = rowobj.get("Opp Losses") if "Opp Losses" in rowobj.index else ""

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{r['Team']}**")
            st.write(f"Coach: {t_coach}")
            st.write(f"Conference: {t_conf}")
            if t_wins is not None and t_losses is not None:
                st.write(f"Record: {t_wins}-{t_losses}")
        with c2:
            st.markdown(f"**{r['Opponent']}**")
            st.write(f"Coach: {o_coach}")
            st.write(f"Conference: {o_conf}")
            if o_wins is not None and o_losses is not None:
                st.write(f"Record: {o_wins}-{o_losses}")

        st.markdown("---")
        st.write("Top-50 Rank Categories")
        left, right = st.columns(2)

        # Because the schedule row is from one perspective (Team vs Opponent),
        # the row's *_Rank fields correspond to the left team perspective.
        # We'll still extract split lists (they will usually be mostly left-leaning),
        # but we keep the split presentation you asked for.
        top50_team = extract_team_top50(rowobj)
        with left:
            st.write(f"**{r['Team']}**")
            if not top50_team:
                st.write("None in top 50")
            else:
                st.dataframe(pd.DataFrame([{"Category": k, "Rank": v} for k, v in top50_team]),
                             use_container_width=True)

        # For opponent, try to map opponent-specific rank columns if present;
        # if not, we’ll do the same extraction (often empty early season, which is fine).
        # Heuristic: try to find any columns that start with 'OPP_' and end with 'RANK' etc.
        opp_cols = {}
        for c in rowobj.index:
            cu = c.upper()
            if ("OPP_" in cu) and ("RANK" in cu or cu.endswith(" RANK") or cu.endswith("_RANK")):
                opp_cols[c] = rowobj.get(c)
        top50_opp = []
        for k, v in opp_cols.items():
            rv = clean_rank_value(v)
            if rv <= 50:
                top50_opp.append((k, int(rv)))
        top50_opp = sorted(top50_opp, key=lambda x: x[1])

        with right:
            st.write(f"**{r['Opponent']}**")
            if not top50_opp:
                st.write("None in top 50")
            else:
                st.dataframe(pd.DataFrame([{"Category": k, "Rank": v} for k, v in top50_opp]),
                             use_container_width=True)

# ------------------------------------------
# EXPORT SLATE
# ------------------------------------------
st.markdown("---")
st.markdown("### Export")
export_cols = ["Date","Team","Opponent","BestMarket","BestLineVal","BestDecOdds","Bet Prob","BestEdge","Stake"]
export_df = slate.rename(columns={"BestProb":"Bet Prob"})[export_cols].copy()
export_df["Date"] = export_df["Date"].dt.strftime("%Y-%m-%d")
csv = export_df.to_csv(index=False)
st.download_button("Download Slate CSV", data=csv, file_name="betting_slate.csv", mime="text/csv")
