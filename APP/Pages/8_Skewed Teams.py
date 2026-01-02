import os
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV

# ----------------------------
# CONFIG (same as Betting page)
# ----------------------------
BASE = "Data/26_March_Madness_Databook"
DAILY_FILE = os.path.join(BASE, "Daily_predictor_data-Table 1.csv")

st.set_page_config(page_title="Team Betting Breakdown", layout="wide")
st.title("Team Betting Breakdown (ML / Spread / OU + Point Error)")

# ----------------------------
# Helpers (minimal)
# ----------------------------
def load_csv(path: str):
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()
    df = pd.read_csv(path, encoding="latin1")
    df.columns = df.columns.str.strip()
    return df

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def parse_mdy(series_like: pd.Series) -> pd.Series:
    s = series_like.astype(str).str.strip()
    out = pd.to_datetime(s, errors="coerce", format="%m/%d/%Y")
    mask = out.isna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(s[mask], errors="coerce", infer_datetime_format=True)
    return out

def clamp_prob(p, lo=0.005, hi=0.995):
    if not np.isfinite(p):
        return np.nan
    return float(min(hi, max(lo, p)))

# ----------------------------
# Load data
# ----------------------------
daily_df = load_csv(DAILY_FILE)

d_team = find_col(daily_df, ["Team","Teams","team"])
d_opp  = find_col(daily_df, ["Opponent","Opp","opponent"])
d_pts  = find_col(daily_df, ["Points"])
d_opp_pts = find_col(daily_df, ["Opp Points","Opp_Points","OppPoints"])
d_line = find_col(daily_df, ["Line"])
d_ou   = find_col(daily_df, ["Over/Under Line","OverUnder","Over Under Line","O/U"])
d_date = find_col(daily_df, ["Date","Game Date","Game_Date"])

if any(x is None for x in [d_team, d_opp, d_pts, d_opp_pts]):
    st.error("Daily predictor must include Team, Opponent, Points, Opp Points.")
    st.stop()

daily_df[d_pts] = pd.to_numeric(daily_df[d_pts], errors="coerce")
daily_df[d_opp_pts] = pd.to_numeric(daily_df[d_opp_pts], errors="coerce")
daily_df["__Date"] = parse_mdy(daily_df[d_date]) if d_date else pd.to_datetime("2000-01-01")

# Targets
daily_df["__SM"] = daily_df[d_pts] - daily_df[d_opp_pts]
daily_df["__TOTAL"] = daily_df[d_pts] + daily_df[d_opp_pts]

y_win = (daily_df[d_pts] > daily_df[d_opp_pts]).astype(int)

y_cover = None
if d_line is not None:
    daily_df[d_line] = pd.to_numeric(daily_df[d_line], errors="coerce")
    y_cover = ((daily_df["__SM"] + daily_df[d_line]) > 0).astype(int)

y_over = None
if d_ou is not None:
    daily_df[d_ou] = pd.to_numeric(daily_df[d_ou], errors="coerce")
    y_over = (daily_df["__TOTAL"] > daily_df[d_ou]).astype(int)

# Train rows only where points exist
mask_points = daily_df[d_pts].notna() & daily_df[d_opp_pts].notna()
df_train = daily_df.loc[mask_points].copy()

# Feature columns: numeric + categorical
numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
# remove pure targets from numeric features
for drop in [d_pts, d_opp_pts, "__SM", "__TOTAL"]:
    if drop in numeric_cols:
        numeric_cols.remove(drop)

cat_candidates = [d_team, d_opp, "Conference","Opponent Conference","HAN","Home/Away","Location"]
cat_cols = [c for c in cat_candidates if c and c in df_train.columns]
cat_cols = list(dict.fromkeys(cat_cols))
numeric_cols = [c for c in numeric_cols if c not in cat_cols]

# Coerce numeric
for c in numeric_cols:
    df_train[c] = pd.to_numeric(df_train[c], errors="coerce")

# Preprocessor
try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

preproc = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_cols),
        ("cat", ohe, cat_cols)
    ],
    remainder="drop",
    sparse_threshold=0.0
)

X = preproc.fit_transform(df_train[numeric_cols + cat_cols])

# Models (keep it similar to your Betting page)
rf_points = RandomForestRegressor(
    n_estimators=900, random_state=42, n_jobs=-1, min_samples_leaf=1, max_features="sqrt"
)
rf_points.fit(X, df_train[[d_pts, d_opp_pts]].values)

base_win = RandomForestClassifier(n_estimators=800, random_state=42, class_weight="balanced_subsample", n_jobs=-1)
clf_win = CalibratedClassifierCV(base_win, method="isotonic", cv=5)
clf_win.fit(X, y_win.loc[df_train.index].values.astype(int))

clf_cover = None
if y_cover is not None:
    y_cov = y_cover.loc[df_train.index].values.astype(float)
    m = ~np.isnan(y_cov)
    if m.any():
        base_cover = RandomForestClassifier(n_estimators=700, random_state=42, class_weight="balanced_subsample", n_jobs=-1)
        clf_cover = CalibratedClassifierCV(base_cover, method="isotonic", cv=5)
        clf_cover.fit(X[m], y_cov[m].astype(int))

clf_over = None
if y_over is not None:
    y_ov = y_over.loc[df_train.index].values.astype(float)
    m = ~np.isnan(y_ov)
    if m.any():
        base_over = RandomForestClassifier(n_estimators=700, random_state=42, class_weight="balanced_subsample", n_jobs=-1)
        clf_over = CalibratedClassifierCV(base_over, method="isotonic", cv=5)
        clf_over.fit(X[m], y_ov[m].astype(int))

# ----------------------------
# Per-row predictions (historical)
# ----------------------------
proba_win = clf_win.predict_proba(X)[:, 1]
pred_win = (proba_win >= 0.5).astype(int)

pts_pred = rf_points.predict(X)
pred_pts = pts_pred[:, 0]
pred_opp_pts = pts_pred[:, 1]
pred_margin = pred_pts - pred_opp_pts
pred_total = pred_pts + pred_opp_pts

df_eval = df_train[[d_team, d_opp, "__Date", d_pts, d_opp_pts, "__SM", "__TOTAL"]].copy()
df_eval["Win_p"] = proba_win
df_eval["Win_pred"] = pred_win

df_eval["Pred_PTS"] = pred_pts
df_eval["Pred_OPP_PTS"] = pred_opp_pts
df_eval["Pred_Margin"] = pred_margin
df_eval["Pred_Total"] = pred_total

df_eval["Err_PTS"] = df_eval["Pred_PTS"] - df_eval[d_pts]
df_eval["Err_OPP_PTS"] = df_eval["Pred_OPP_PTS"] - df_eval[d_opp_pts]
df_eval["Err_Margin"] = df_eval["Pred_Margin"] - df_eval["__SM"]
df_eval["Err_Total"] = df_eval["Pred_Total"] - df_eval["__TOTAL"]

# Spread / OU
if d_line is not None and clf_cover is not None:
    proba_cov = clf_cover.predict_proba(X)[:, 1]
    df_eval["Cover_p"] = proba_cov
    df_eval["Cover_pred"] = (proba_cov >= 0.5).astype(int)
    df_eval["Cover_y"] = y_cover.loc[df_train.index].values.astype(int)
else:
    df_eval["Cover_p"] = np.nan
    df_eval["Cover_pred"] = np.nan
    df_eval["Cover_y"] = np.nan

if d_ou is not None and clf_over is not None:
    proba_over = clf_over.predict_proba(X)[:, 1]
    df_eval["Over_p"] = proba_over
    df_eval["Over_pred"] = (proba_over >= 0.5).astype(int)
    df_eval["Over_y"] = y_over.loc[df_train.index].values.astype(int)
else:
    df_eval["Over_p"] = np.nan
    df_eval["Over_pred"] = np.nan
    df_eval["Over_y"] = np.nan

# ----------------------------
# Team summary table
# ----------------------------
def brier(p, y):
    m = np.isfinite(p) & np.isfinite(y)
    if m.sum() == 0:
        return np.nan
    return float(np.mean((p[m] - y[m])**2))

team_grp = df_eval.groupby(d_team)

summary = team_grp.agg(
    Games=("Win_pred", "size"),
    ML_Acc=("Win_pred", lambda s: float(np.mean(s.values == df_eval.loc[s.index, "Win_pred"].values))),  # placeholder
)

# Recompute accuracies cleanly
rows = []
for team, g in team_grp:
    ml_acc = float(np.mean(g["Win_pred"].values == (g[d_pts].values > g[d_opp_pts].values).astype(int)))
    ml_brier = brier(g["Win_p"].values, (g[d_pts].values > g[d_opp_pts].values).astype(int))

    cover_acc = np.nan
    cover_brier = np.nan
    if np.isfinite(g["Cover_pred"].values).any():
        m = np.isfinite(g["Cover_pred"].values) & np.isfinite(g["Cover_y"].values)
        if m.sum() > 0:
            cover_acc = float(np.mean(g.loc[m, "Cover_pred"].values == g.loc[m, "Cover_y"].values))
            cover_brier = brier(g.loc[m, "Cover_p"].values, g.loc[m, "Cover_y"].values)

    ou_acc = np.nan
    ou_brier = np.nan
    if np.isfinite(g["Over_pred"].values).any():
        m = np.isfinite(g["Over_pred"].values) & np.isfinite(g["Over_y"].values)
        if m.sum() > 0:
            ou_acc = float(np.mean(g.loc[m, "Over_pred"].values == g.loc[m, "Over_y"].values))
            ou_brier = brier(g.loc[m, "Over_p"].values, g.loc[m, "Over_y"].values)

    rows.append({
        "Team": team,
        "Games": len(g),
        "ML_Acc": ml_acc,
        "ML_Brier": ml_brier,
        "Spread_Acc": cover_acc,
        "Spread_Brier": cover_brier,
        "OU_Acc": ou_acc,
        "OU_Brier": ou_brier,
        "Margin_Bias": float(np.mean(g["Err_Margin"])),
        "Margin_MAE": float(np.mean(np.abs(g["Err_Margin"]))),
        "Total_Bias": float(np.mean(g["Err_Total"])),
        "Total_MAE": float(np.mean(np.abs(g["Err_Total"]))),
    })

summary_df = pd.DataFrame(rows).sort_values("Games", ascending=False)

# Flag “sway” teams (example thresholds)
margin_mae_90 = np.nanpercentile(summary_df["Margin_MAE"].values, 90)
total_mae_90 = np.nanpercentile(summary_df["Total_MAE"].values, 90)

summary_df["FLAG_Sway"] = (
    (summary_df["Margin_MAE"] >= margin_mae_90) |
    (summary_df["Total_MAE"] >= total_mae_90) |
    (summary_df["ML_Acc"] <= (summary_df["ML_Acc"].median() - 0.08))
)

# ----------------------------
# UI
# ----------------------------
st.subheader("League-wide Team Table")
st.dataframe(summary_df, use_container_width=True)

st.subheader("Single Team Drilldown")
team_pick = st.selectbox("Select team", sorted(df_eval[d_team].unique()))
g = df_eval[df_eval[d_team] == team_pick].copy().sort_values("__Date", ascending=False)

c1, c2, c3, c4 = st.columns(4)
c1.metric("ML Accuracy", f"{float(np.mean(g['Win_pred'].values == (g[d_pts].values > g[d_opp_pts].values).astype(int))):.3f}")
c2.metric("Margin MAE", f"{float(np.mean(np.abs(g['Err_Margin']))):.2f}")
c3.metric("Total MAE", f"{float(np.mean(np.abs(g['Err_Total']))):.2f}")
c4.metric("Margin Bias", f"{float(np.mean(g['Err_Margin'])):+.2f}")

st.write("Recent games (model vs actual)")
show_cols = [
    "__Date", d_opp, d_pts, d_opp_pts,
    "Pred_PTS","Pred_OPP_PTS","Err_Margin","Err_Total","Win_p"
]
st.dataframe(g[show_cols].head(25), use_container_width=True)
