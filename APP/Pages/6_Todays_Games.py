import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data/26_March_Madness_Databook/2026 Schedule Transfer-Table 1.csv", encoding="latin1")
    df = df.fillna(0)
    # Ensure proper date ordering (YYYY-MM-DD)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", infer_datetime_format=True)
    df = df.sort_values("Date")
    return df

df = load_data()

# ----------------------------
# PREP ML MODEL
# ----------------------------
st.title("🏀 Today's Games - Predictive Scoreboard")

# Only train on rows with actual scores
train_df = df[df["Points"].notna() & df["Opp Points"].notna()]
if train_df.empty:
    st.warning("Not enough completed games to train the prediction model yet.")
else:
    # Use all numeric columns except target
    X = train_df.select_dtypes(include=[np.number]).drop(columns=["Points", "Opp Points"], errors="ignore")
    y_points = train_df["Points"]
    y_opp_points = train_df["Opp Points"]

    # Train models
    X_train, X_test, y_train, y_test = train_test_split(X, y_points, test_size=0.2, random_state=42)
    model_points = RandomForestRegressor(n_estimators=200, random_state=42)
    model_points.fit(X_train, y_train)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_opp_points, test_size=0.2, random_state=42)
    model_opp = RandomForestRegressor(n_estimators=200, random_state=42)
    model_opp.fit(X_train2, y_train2)

    # ----------------------------
    # PREDICT UPCOMING GAMES
    # ----------------------------
    future_games = df[df["Points"] == 0].copy()
    if not future_games.empty:
        X_future = future_games.select_dtypes(include=[np.number]).drop(columns=["Points", "Opp Points"], errors="ignore")

        future_games["Pred_Points"] = model_points.predict(X_future)
        future_games["Pred_Opp_Points"] = model_opp.predict(X_future)

        # ----------------------------
        # RANKING LOGIC FOR DISPLAY
        # ----------------------------
        def game_priority(row):
            if row["Top 25 Opponent"] == 1:
                return 1
            elif row["March Madness Opponent"] == 1:
                return 2
            elif row["Conference"] in ["SEC", "Big Ten", "Big 12", "ACC"] or row["Opponent Conference"] in ["SEC", "Big Ten", "Big 12", "ACC"]:
                return 3
            else:
                return 4

        future_games["Priority"] = future_games.apply(game_priority, axis=1)
        future_games = future_games.sort_values(["Priority", "Date"])

        # ----------------------------
        # DISPLAY EACH GAME
        # ----------------------------
        for _, row in future_games.iterrows():
            game_label = f"**{row['Team']} vs {row['Opponent']}** — {row['Date'].strftime('%b %d, %Y')}"
            pred_score = f"🧮 *Projected:* {row['Team']} {row['Pred_Points']:.1f} — {row['Opponent']} {row['Pred_Opp_Points']:.1f}"

            with st.expander(f"{game_label}\n{pred_score}"):
                col1, col2 = st.columns(2)

                # TEAM INFO TABLE
                team_info = {
                    "Conference": row.get("Conference", ""),
                    "Coach": row.get("Coach Name", ""),
                    "Wins": row.get("Wins", 0),
                    "Losses": row.get("Losses", 0)
                }
                opp_info = {
                    "Conference": row.get("Opponent Conference", ""),
                    "Coach": row.get("Opponent Coach", ""),
                    "Wins": row.get("Wins", 0),  # Placeholder if opponent has separate stats later
                    "Losses": row.get("Losses", 0)
                }

                with col1:
                    st.markdown(f"### {row['Team']}")
                    st.table(pd.DataFrame(team_info, index=[0]).T)

                with col2:
                    st.markdown(f"### {row['Opponent']}")
                    st.table(pd.DataFrame(opp_info, index=[0]).T)

                # ----------------------------
                # TOP 50 RANKS
                # ----------------------------
                rank_cols = [c for c in df.columns if any(s in c for s in ["_Rank", "_rank", "Rank"])]

                team_top50 = {}
                opp_top50 = {}
                for c in rank_cols:
                    try:
                        val_team = float(row.get(c, np.nan))
                        val_opp = float(row.get(c.replace("OPP_", ""), np.nan))
                        if val_team <= 50:
                            team_top50[c] = int(val_team)
                        if val_opp <= 50:
                            opp_top50[c] = int(val_opp)
                    except:
                        continue

                if team_top50 or opp_top50:
                    st.markdown("---")
                    st.markdown("### 🏅 Top 50 National Categories")
                    combined = pd.DataFrame([
                        {"Category": k, row["Team"]: v, row["Opponent"]: opp_top50.get(k, "")}
                        for k, v in team_top50.items()
                    ])
                    if combined.empty:
                        st.info("No Top 50 categories for this matchup yet.")
                    else:
                        st.dataframe(combined, use_container_width=True)

    else:
        st.info("No upcoming games found.")
