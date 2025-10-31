# APP/Pages/7_Betting.py
import streamlit as st
import pandas as pd
import numpy as np
import os, math
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OrdinalEncoder

# ------------------------------------------
# CONFIG
# ------------------------------------------
BASE = "Data/26_March_Madness_Databook"
SCHEDULE_FILE = os.path.join(BASE, "2026 Schedule Transfer-Table 1.csv")
DAILY_FILE = os.path.join(BASE, "Daily_predictor_data-Table 1.csv")

st.set_page_config(page_title="Betting (Private)", layout="wide")
st.title("Betting (Private)")

# ------------------------------------------
# LOADERS
# ------------------------------------------
@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, encoding="latin1")
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return None

schedule_df = load_csv(SCHEDULE_FILE)
daily_df = load_csv(DAILY_FILE)

if schedule_df is None:
    st.error(f"Could not find file: {SCHEDULE_FILE}")
    st.stop()
if daily_df is None:
    st.error(f"Could not find file: {DAILY_FILE}")
    st.stop()

# ------------------------------------------
# UTILS
# ------------------------------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def parse_mmddyyyy_series(series):
    # Try strict MM/DD/YYYY first; then fall back to generic parser.
    parsed = pd.to_datetime(series, format="%m/%d/%Y", errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(series.astype(str).str.strip(), errors="coerce")
    return parsed

def interpret_han(v):
    if pd.isna(v): return None
    s = str(v).strip().upper()
    if s in ("H","HOME"): return "Home"
    if s in ("A","AWAY"): return "Away"
    if s in ("N","NEUTRAL") or "NEUTRAL" in s: return "Neutral"
    return None

# Odds helpers
def american_to_decimal(odds):
    if odds is None or odds == "" or pd.isna(odds):
        return None
    try:
        o = float(odds)
    except:
        return None
    if o > 0:
        return 1.0 + (o/100.0)
    elif o < 0:
        return 1.0 + (100.0/abs(o))
    else:
        return None

def implied_prob_from_american(odds):
    if odds is None or odds == "" or pd.isna(odds):
        return None
    try:
        o = float(odds)
    except:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    elif o < 0:
        return abs(o) / (abs(o) + 100.0)
    else:
        return None

def kelly_fraction(p, dec_odds):
    # p: win prob (0-1); dec_odds: decimal odds (>1)
    # kelly = (p*(b) - (1-p)) / b where b = dec_odds - 1
    if p is None or dec_odds is None or dec_odds <= 1.0:
        return 0.0
    b = dec_odds - 1.0
    k = (p*b - (1.0 - p)) / b
    return max(0.0, k)

def normal_cdf(x):
    # Approx via math.erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# ------------------------------------------
# COLUMN DETECTION
# ------------------------------------------
team_col = find_col(schedule_df, ["Team","Teams"])
opp_col = find_col(schedule_df, ["Opponent","Opp"])
date_col = find_col(schedule_df, ["Date","date"])
han_col = find_col(schedule_df, ["HAN","Home/Away","Location","Loc","HomeAway","HOME/AWAY"])
line_col = find_col(schedule_df, ["Line","Spread","line"])
ml_col = find_col(schedule_df, ["ML","Moneyline","moneyline"])
ou_col = find_col(schedule_df, ["Over/Under Line","OverUnder","Total","O/U"])

coach_col = find_col(schedule_df, ["Coach Name","Coach","Coach_Name"])
opp_coach_col = find_col(schedule_df, ["Opponent Coach","Opp Coach"])
conf_col = find_col(schedule_df, ["Conference","Conf"])
opp_conf_col = find_col(schedule_df, ["Opponent Conference","Opp Conference","Opp_Conference"])

wins_col = find_col(schedule_df, ["Wins","W"])
losses_col = find_col(schedule_df, ["Losses","L"])

top25_col = find_col(schedule_df, ["Top 25 Opponent","Top25"])
mm_col = find_col(schedule_df, ["March Madness Opponent","March_Madness"])

# Training targets from daily
d_team_col = find_col(daily_df, ["Team","Teams"])
d_opp_col  = find_col(daily_df, ["Opponent","Opp"])
d_pts_col  = find_col(daily_df, ["Points","PTS"])
d_opppts_col = find_col(daily_df, ["Opp Points","Opp_Points","OPP_PTS"])

if any(c is None for c in [team_col, opp_col, date_col, d_team_col, d_opp_col, d_pts_col, d_opppts_col]):
    st.error("Missing required columns (team/opponent/date in schedule or team/opponent/points in daily).")
    st.stop()

# ------------------------------------------
# DATE & FILTERS
# ------------------------------------------
schedule_df["__Date"] = parse_mmddyyyy_series(schedule_df[date_col])
schedule_df = schedule_df.dropna(subset=["__Date"]).reset_index(drop=True)

# Page filters
left, right = st.columns([2,1])
with left:
    date_choice = st.date_input("Filter by date (optional)", value=None)
with right:
    only_with_lines = st.checkbox("Show only games with any available market (Line / ML / O/U)", value=True)

if date_choice:
    schedule_df = schedule_df.loc[schedule_df["__Date"].dt.date == date_choice]

if schedule_df.empty:
    st.info("No games after applying date filter.")
    st.stop()

# ------------------------------------------
# TRAIN MODELS (multi-output points + win classifier)
# ------------------------------------------
# Build features from daily using schedule-style columns (team/opp + categorical context).
# For numeric features: use *all numeric* columns present in daily (except the two targets).
d_numeric_cols = daily_df.select_dtypes(include=[np.number]).columns.tolist()
d_numeric_cols = [c for c in d_numeric_cols if c not in (d_pts_col, d_opppts_col)]

# Categorical columns we’ll encode (robust to availability)
cat_cols = []
for c in [d_team_col, d_opp_col, "Coach Name", "Opponent Coach", "Conference", "Opponent Conference", "HAN"]:
    if isinstance(c, str) and c in daily_df.columns:
        cat_cols.append(c)

# Build training X/Y
# Safe numeric block
X_num = daily_df[d_numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values if d_numeric_cols else np.zeros((len(daily_df), 0))
Y_points = np.column_stack([
    pd.to_numeric(daily_df[d_pts_col], errors="coerce").values,
    pd.to_numeric(daily_df[d_opppts_col], errors="coerce").values
])

# Filter rows that have both targets
valid_mask = ~np.isnan(Y_points).any(axis=1)
X_num = X_num[valid_mask]
Y_points = Y_points[valid_mask]

# Encode categoricals (ordinal is fine for RF)
enc = None
X_cat = np.zeros((X_num.shape[0], 0))
if cat_cols:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_matrix = daily_df.loc[valid_mask, cat_cols].fillna("NA").astype(str).values
    try:
        X_cat = enc.fit_transform(cat_matrix)
    except Exception:
        X_cat = np.zeros((X_num.shape[0], len(cat_cols)))

# Final training features
X_train = np.hstack([X_num, X_cat]) if X_cat.shape[1] > 0 else X_num

# If we also want a win classifier
Y_win = (Y_points[:,0] > Y_points[:,1]).astype(int)

# Fit models
if X_train.shape[0] < 20:
    st.warning(f"Only {X_train.shape[0]} rows with targets in Daily; bets will be noisy.")

rf_multi = MultiOutputRegressor(RandomForestRegressor(n_estimators=500, random_state=42))
rf_multi.fit(X_train, Y_points)

clf = RandomForestClassifier(n_estimators=400, random_state=42)
clf.fit(X_train, Y_win)

# Residual std estimates for cover/total probability approximations
train_pred = rf_multi.predict(X_train)
margin_resid = (Y_points[:,0] - Y_points[:,1]) - (train_pred[:,0] - train_pred[:,1])
total_resid  = (Y_points[:,0] + Y_points[:,1]) - (train_pred[:,0] + train_pred[:,1])
sd_margin = float(np.std(margin_resid)) if margin_resid.size else 10.0
sd_total  = float(np.std(total_resid))  if total_resid.size  else 20.0

# ------------------------------------------
# BUILD PREDICTION FEATURES FOR SCHEDULE ROWS
# ------------------------------------------
# Align schedule numeric features to daily’s numeric feature set (by exact name match);
# missing ones become 0. Categorical via the same enc mapping (unknown -> -1).
def row_to_feature_vector(row):
    # numeric
    if d_numeric_cols:
        vals = []
        for c in d_numeric_cols:
            v = row.get(c, 0)
            try:
                vals.append(float(v) if (pd.notna(v) and v != "") else 0.0)
            except:
                vals.append(0.0)
        Xn = np.array(vals, dtype=float).reshape(1, -1)
    else:
        Xn = np.zeros((1,0))
    # cats
    if enc and cat_cols:
        arr = []
        for c in cat_cols:
            arr.append(str(row.get(c, "NA")))
        try:
            Xc = enc.transform([arr])
        except Exception:
            Xc = np.array([[-1]*len(cat_cols)], dtype=float)
        Xf = np.hstack([Xn, Xc])
    else:
        Xf = Xn
    if Xf.shape[1] != X_train.shape[1]:
        # pad/truncate to training width
        out = np.zeros((1, X_train.shape[1]), dtype=float)
        m = min(Xf.shape[1], X_train.shape[1])
        out[0,:m] = Xf[0,:m]
        return out
    return Xf

# Predict for each schedule row; keep only rows with any market if requested
working = schedule_df.copy()

if only_with_lines:
    def has_any_market(r):
        return (not pd.isna(r.get(line_col))) or (not pd.isna(r.get(ml_col))) or (not pd.isna(r.get(ou_col)))
    working = working.loc[working.apply(has_any_market, axis=1)]

if working.empty:
    st.info("No games with available markets.")
    st.stop()

# Ensure required categorical columns exist in schedule for feature building
for c in cat_cols:
    if c not in working.columns:
        working[c] = "NA"
# Align obvious schedule cat cols if names differ
if "HAN" in cat_cols and (han_col and han_col != "HAN"):
    working["HAN"] = working[han_col].apply(interpret_han).fillna("NA")

# Force numeric coercion for markets
def to_float_or_none(x):
    try:
        return float(x)
    except:
        return None

working["__Line"] = working[line_col].apply(to_float_or_none) if line_col else None
working["__ML"]   = working[ml_col].apply(to_float_or_none)   if ml_col   else None
working["__OU"]   = working[ou_col].apply(to_float_or_none)   if ou_col   else None

# Predictions
pred_points = []
pred_opp_points = []
win_probs = []

for _, r in working.iterrows():
    xf = row_to_feature_vector(r)
    pts_pair = rf_multi.predict(xf)[0]
    p_win = float(clf.predict_proba(xf)[0][1])
    pred_points.append(float(max(0.0, pts_pair[0])))
    pred_opp_points.append(float(max(0.0, pts_pair[1])))
    win_probs.append(p_win)

working["Pred_Points"] = pred_points
working["Pred_Opp_Points"] = pred_opp_points
working["Pred_Spread"] = working["Pred_Points"] - working["Pred_Opp_Points"]  # Team perspective
working["Pred_Total"] = working["Pred_Points"] + working["Pred_Opp_Points"]
working["Win_Prob"] = win_probs

# ------------------------------------------
# USER STRATEGY CONTROLS
# ------------------------------------------
st.markdown("### Controls")

c1,c2,c3 = st.columns(3)
with c1:
    unit_bankroll = st.number_input("Units of betting (bankroll for today)", min_value=1.0, value=100.0, step=1.0)
with c2:
    floor_units = st.number_input("Units floor for a bad day (95% conf goal)", min_value=0.0, value=65.5, step=0.5)
with c3:
    hope_units = st.number_input("Units target for a solid day (~70% hit)", min_value=0.0, value=140.0, step=1.0)

c4,c5,c6 = st.columns(3)
with c4:
    homerun = st.slider("Homerun Hitter (parlay aggression)", min_value=1, max_value=10, value=7)
with c5:
    ladder_start = st.checkbox("Ladder Starter", value=False)
with c6:
    ladder_cont = st.checkbox("Ladder Continuation", value=False)

ladder_units = 0.0
if ladder_cont:
    ladder_units = st.number_input("Ladder Continuation units (reserved from bankroll)", min_value=0.0, value=10.0, step=1.0)

min_bets, max_bets = st.slider("Number of bets (min/max)", min_value=4, max_value=100, value=(8, 20), step=1)

# Adjusted bankroll for non-ladder bets
avail_units = max(0.0, unit_bankroll - ladder_units)

# Scale Kelly by “risk” implied by floor/hope + homerun
# Simple mapping: risk_scale in [0.25..1.25]
risk_scale = 0.75 + 0.05*(homerun - 5)
risk_scale = max(0.25, min(1.25, risk_scale))

# ------------------------------------------
# BUILD CANDIDATE BETS (Sides / Totals / Moneylines)
# ------------------------------------------
cands = []

# Sides (spread):
# Probability to cover using normal approx around Pred_Spread with sd_margin
# P(Team -line covers) = 1 - CDF((line - pred_margin)/sd)
for idx, r in working.iterrows():
    team = r[team_col]; opp = r[opp_col]
    line = r["__Line"]
    if line is not None:
        z = (line - r["Pred_Spread"]) / (sd_margin if sd_margin>1e-6 else 8.0)
        p_cover = 1.0 - normal_cdf(z)
        # Approx American odds for spread near -110 each side if line exists but no separate price
        dec_odds = american_to_decimal(-110)
        k = kelly_fraction(p_cover, dec_odds) * risk_scale
        ev_per_unit = p_cover*(dec_odds-1) - (1-p_cover)
        cands.append({
            "Type":"Spread", "Team":team, "Opponent":opp, "Market":f"{team} {line:+.1f}",
            "Prob":p_cover, "DecOdds":dec_odds, "American":-110, "EV_per_unit":ev_per_unit,
            "KellyFrac":k, "GameIndex":idx
        })
        # Opposite side
        z2 = (-line - r["Pred_Spread"]) / (sd_margin if sd_margin>1e-6 else 8.0)
        p_cover_opp = 1.0 - normal_cdf(z2)
        ev_per_unit_opp = p_cover_opp*(dec_odds-1) - (1-p_cover_opp)
        cands.append({
            "Type":"Spread", "Team":opp, "Opponent":team, "Market":f"{opp} {-line:+.1f}",
            "Prob":p_cover_opp, "DecOdds":dec_odds, "American":-110, "EV_per_unit":ev_per_unit_opp,
            "KellyFrac":kelly_fraction(p_cover_opp, dec_odds)*risk_scale, "GameIndex":idx
        })

# Totals
for idx, r in working.iterrows():
    ou = r["__OU"]
    if ou is not None:
        z_over = (ou - r["Pred_Total"]) / (sd_total if sd_total>1e-6 else 16.0)
        p_over = 1.0 - normal_cdf(z_over)
        dec_odds = american_to_decimal(-110)
        ev_over = p_over*(dec_odds-1) - (1-p_over)
        cands.append({
            "Type":"Total", "Team":r[team_col], "Opponent":r[opp_col], "Market":f"Over {ou:.1f}",
            "Prob":p_over, "DecOdds":dec_odds, "American":-110, "EV_per_unit":ev_over,
            "KellyFrac":kelly_fraction(p_over, dec_odds)*risk_scale, "GameIndex":idx
        })
        p_under = 1.0 - p_over
        ev_under = p_under*(dec_odds-1) - (1-p_under)
        cands.append({
            "Type":"Total", "Team":r[team_col], "Opponent":r[opp_col], "Market":f"Under {ou:.1f}",
            "Prob":p_under, "DecOdds":dec_odds, "American":-110, "EV_per_unit":ev_under,
            "KellyFrac":kelly_fraction(p_under, dec_odds)*risk_scale, "GameIndex":idx
        })

# Moneylines
for idx, r in working.iterrows():
    ml = r["__ML"]
    if ml is not None:
        dec = american_to_decimal(ml)
        p = r["Win_Prob"]  # model win prob for the schedule "Team"
        ev = p*(dec-1) - (1-p)
        cands.append({
            "Type":"Moneyline", "Team":r[team_col], "Opponent":r[opp_col], "Market":f"{r[team_col]} ML ({int(ml):+d})",
            "Prob":p, "DecOdds":dec, "American":ml, "EV_per_unit":ev,
            "KellyFrac":kelly_fraction(p, dec)*risk_scale, "GameIndex":idx
        })

cand_df = pd.DataFrame(cands)
if cand_df.empty:
    st.info("No candidate bets found (no markets available).")
    st.stop()

# ------------------------------------------
# RANK & SIZE BETS
# ------------------------------------------
# Rank by EV per unit; cap Kelly; convert to stake in units; enforce bankroll and count range
cand_df = cand_df.sort_values("EV_per_unit", ascending=False).reset_index(drop=True)

# Cap Kelly between [0, 0.2] scaled by risk
kelly_cap = 0.2 * risk_scale
cand_df["StakeFrac"] = cand_df["KellyFrac"].clip(lower=0.0, upper=kelly_cap)
cand_df["StakeUnits_raw"] = cand_df["StakeFrac"] * avail_units * 0.1  # throttle overall risk (10% of bankroll baseline)
# Minimum meaningful stake: 0.25 units; round to 0.1 units
cand_df["StakeUnits"] = (cand_df["StakeUnits_raw"].clip(lower=0.25)).round(1)

# Keep unique bets per game+market; limit count to [min_bets, max_bets] within bankroll
selected = []
units_used = 0.0
for _, row in cand_df.iterrows():
    if len(selected) >= max_bets:
        break
    if units_used + row["StakeUnits"] > avail_units:
        continue
    selected.append(row)
    units_used += row["StakeUnits"]

# If below min_bets, top-up with smallest stakes
if len(selected) < min_bets:
    need = min_bets - len(selected)
    remaining = cand_df.loc[~cand_df.index.isin([r.name for r in selected])].head(need)
    for _, row in remaining.iterrows():
        if len(selected) >= max_bets:
            break
        # force a small 0.25u try if budget allows
        if units_used + 0.25 <= avail_units:
            row2 = row.copy()
            row2["StakeUnits"] = 0.25
            selected.append(row2)
            units_used += 0.25

bets_df = pd.DataFrame(selected)
if bets_df.empty:
    st.info("No bets could be sized under the bankroll/limits chosen.")
    st.stop()

# ------------------------------------------
# LADDER HANDLING
# ------------------------------------------
# Define a “lock-ish” finder: highest win prob single with odds in [-150,+150]
def pick_ladder_candidate(exclude_gameindexes=set()):
    singles = cand_df[(cand_df["Type"].isin(["Moneyline","Spread"]))].copy()
    singles["is_fair_odds"] = singles["American"].apply(lambda a: a is not None and -150 <= float(a) <= 150) if "American" in singles else False
    singles = singles[singles["is_fair_odds"] == True]
    if not singles.empty:
        singles = singles.sort_values(["Prob","EV_per_unit"], ascending=[False, False])
        for _, r in singles.iterrows():
            if r["GameIndex"] not in exclude_gameindexes:
                return r
    # fallback: best overall prob regardless of odds
    singles = cand_df[cand_df["Type"].isin(["Moneyline","Spread"])].sort_values(["Prob","EV_per_unit"], ascending=[False,False])
    for _, r in singles.iterrows():
        if r["GameIndex"] not in exclude_gameindexes:
            return r
    return None

ladder_notes = []
ladder_bets = []
used_gidx = set()

if ladder_start:
    r = pick_ladder_candidate()
    if r is not None:
        ladder_bets.append({"Label":"Ladder Starter", **r.to_dict(), "StakeUnits": max(1.0, round(avail_units*0.02,1))})
        used_gidx.add(r["GameIndex"])

if ladder_cont and ladder_units > 0:
    r2 = pick_ladder_candidate(exclude_gameindexes=used_gidx)
    if r2 is not None:
        ladder_bets.append({"Label":"Ladder Continuation", **r2.to_dict(), "StakeUnits": round(ladder_units,1)})
        used_gidx.add(r2["GameIndex"])

# Ensure ladder bets are not duplicated in main bet list
if ladder_bets:
    ladder_idx = {b["GameIndex"] for b in ladder_bets}
    bets_df = bets_df[~bets_df["GameIndex"].isin(ladder_idx)].reset_index(drop=True)

# ------------------------------------------
# PARLAY SUGGESTIONS (driven by Homerun scale)
# ------------------------------------------
# We’ll take high-probability edges and build 2–4 leg options. More aggressive => longer average legs.
def decimal_odds_from_row(row):
    if row["Type"] == "Moneyline":
        return row["DecOdds"]
    # For spreads/totals priced -110
    return american_to_decimal(-110)

def suggest_parlays(pool_df, aggression=7, max_count=3):
    if pool_df.empty:
        return []
    # choose legs: 2 legs if aggression <=4; 3 legs if 5-7; 4 legs if 8-10
    if aggression <= 4:
        legs = 2
    elif aggression <= 7:
        legs = 3
    else:
        legs = 4
    # sort by Prob then EV
    pool = pool_df.sort_values(["Prob","EV_per_unit"], ascending=[False, False]).copy()
    parlays = []
    used_games = set()
    # Build up to max_count unique-game parlays
    for k in range(max_count):
        legs_rows = []
        games_used = set()
        for _, r in pool.iterrows():
            if len(legs_rows) >= legs: break
            if r["GameIndex"] in games_used: continue
            legs_rows.append(r)
            games_used.add(r["GameIndex"])
        if len(legs_rows) < legs: break
        # compute combined decimal odds and hit prob (independent approximation)
        dec = 1.0
        p = 1.0
        text_legs = []
        for r in legs_rows:
            d = decimal_odds_from_row(r)
            if d is None: d = american_to_decimal(-110)
            dec *= d
            p *= r["Prob"]
            text_legs.append(f"{r['Type']} • {r['Market']}")
        parlays.append({
            "Legs": legs,
            "Description": " | ".join(text_legs),
            "HitProb": p,
            "DecOdds": dec,
            "American": None,  # can compute if needed
            "SuggestedUnits": round(max(0.25, avail_units * 0.01 * aggression/10.0), 2)
        })
        # rotate pool to avoid identical next parlay
        pool = pool.iloc[legs:].copy()
        if pool.empty: break
    return parlays

parlays = suggest_parlays(bets_df, aggression=homerun, max_count=3)

# ------------------------------------------
# DISPLAY
# ------------------------------------------
st.markdown("### Recommended Bets")
show_cols = ["Type","Team","Opponent","Market","Prob","American","EV_per_unit","StakeUnits"]
display_bets = bets_df.copy()
display_bets["Prob"] = (display_bets["Prob"]*100.0).round(1)
display_bets["EV_per_unit"] = display_bets["EV_per_unit"].round(3)
display_bets = display_bets[show_cols].reset_index(drop=True)
st.dataframe(display_bets, use_container_width=True)

st.write(f"Units allocated (excluding ladder continuation): **{round(bets_df['StakeUnits'].sum(),2)} / {avail_units}**")

if ladder_bets:
    st.markdown("---")
    st.markdown("### Ladder Bets")
    ladder_disp = pd.DataFrame([{
        "Label": b["Label"],
        "Type": b["Type"],
        "Team": b["Team"],
        "Opponent": b["Opponent"],
        "Market": b["Market"],
        "Prob(%)": round(b["Prob"]*100,1),
        "American": b["American"],
        "StakeUnits": b["StakeUnits"]
    } for b in ladder_bets])
    st.dataframe(ladder_disp, use_container_width=True)

if parlays:
    st.markdown("---")
    st.markdown("### Parlay Suggestions")
    par_disp = pd.DataFrame([{
        "Legs": p["Legs"],
        "Legs Description": p["Description"],
        "Hit Prob (%)": round(p["HitProb"]*100,1),
        "Decimal Odds": round(p["DecOdds"],2),
        "Suggested Units": p["SuggestedUnits"]
    } for p in parlays])
    st.dataframe(par_disp, use_container_width=True)

# CSV export
st.markdown("---")
out_cols = ["__Date", team_col, opp_col, "Type","Market","Prob","American","EV_per_unit","StakeUnits"]
export_df = bets_df.copy()
export_df["__Date"] = working.loc[export_df["GameIndex"], "__Date"].dt.strftime("%Y-%m-%d").values
export_df["Prob"] = export_df["Prob"].round(4)
export_df["EV_per_unit"] = export_df["EV_per_unit"].round(4)
export_df = export_df.merge(working[[team_col, opp_col]], left_on="GameIndex", right_index=True, how="left")
export_df = export_df[out_cols]
csv = export_df.to_csv(index=False)
st.download_button("Download Bets CSV", csv, "bets_recommendations.csv", "text/csv")

# Footnotes
st.markdown("---")
st.caption(
    "Notes:\n"
    "- Models retrain from Daily Predictor on each run (multi-output RF for points; RF classifier for win probability).\n"
    "- Spread/total cover probabilities are approximated from the model margin/total vs. market using a normal CDF with residual-based σ.\n"
    "- Kelly stakes are scaled by your Homerun slider and throttled to keep risk reasonable. You can tune bankroll and min/max bets above.\n"
    "- Ladder continuation units are reserved from the bankroll and ladder bets never repeat the same game."
)

