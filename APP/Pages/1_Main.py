import streamlit as st
import pandas as pd

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(page_title="March Metrics", layout="wide")

# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "Data/26_March_Madness_Databook/All_Stats-THE_TABLE.csv",
        encoding="latin1"
    )
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# -------------------------------------------------------------
# PAGE HEADER
# -------------------------------------------------------------
# Centered title styling
st.markdown(
    """
    <h1 style='text-align: center; color: black;'>
        March Metrics
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align: center; font-size: 16px; color: gray; margin-top: -10px;'>
        A data-driven college basketball analytics platform
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

st.write(
    """
**March Metrics** integrates **statistical modeling**, **machine learning**, and **contextual metrics**
to measure team and player performance beyond traditional box scores.

This page provides a snapshot of team strength across Division I programs,  
based on quantitative indicators of efficiency, opponent difficulty, and coaching value.  
All models are built using historical datasets and context-adjusted variables 
to support predictive decision-making during the regular season and postseason.
"""
)

# -------------------------------------------------------------
# TOP 10 TEAMS BY STATISTICAL STRENGTH
# -------------------------------------------------------------
st.markdown("---")
st.subheader("Top 10 Teams by Statistical Strength")

# Standardize and clean columns
rename_map = {
    "STAT_STREN": "Statistical Strength",
    "Coach Name": "Coach Name",
    "Conference": "Conference",
    "Teams": "Teams",
    "Wins": "Wins",
    "Losses": "Losses"
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

df["Statistical Strength"] = pd.to_numeric(df["Statistical Strength"], errors="coerce")

top10 = (
    df[["Teams", "Coach Name", "Conference", "Wins", "Losses", "Statistical Strength"]]
    .sort_values(by="Statistical Strength", ascending=True)
    .head(10)
    .reset_index(drop=True)
)
top10.index = top10.index + 1

st.dataframe(
    top10.style.format({
        "Statistical Strength": "{:.3f}",
        "Wins": "{:.0f}",
        "Losses": "{:.0f}"
    }).background_gradient(subset=["Statistical Strength"], cmap="Greens_r"),
    use_container_width=True,
)

# -------------------------------------------------------------
# SECTION PREVIEWS
# -------------------------------------------------------------
st.markdown("---")
st.subheader("Explore Other Sections")

st.write(
    """
Each module in March Metrics focuses on a different layer of basketball intelligence,  
ranging from team analytics to player modeling and betting projections.
"""
)

preview_cols = st.columns(4)

with preview_cols[0]:
    st.markdown(
        """
        <div style="padding:15px; border-radius:12px; background-color:#E8F8F5;
                    box-shadow:0px 4px 8px rgba(0,0,0,0.1); text-align:left; color:black;">
            <h4>Team Breakdown</h4>
            <p>Advanced efficiency metrics, contextualized by opponent and location.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with preview_cols[1]:
    st.markdown(
        """
        <div style="padding:15px; border-radius:12px; background-color:#FEF9E7;
                    box-shadow:0px 4px 8px rgba(0,0,0,0.1); text-align:left; color:black;">
            <h4>Team Comparison</h4>
            <p>Side-by-side evaluation of multiple teams using predictive indicators.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with preview_cols[2]:
    st.markdown(
        """
        <div style="padding:15px; border-radius:12px; background-color:#FDEDEC;
                    box-shadow:0px 4px 8px rgba(0,0,0,0.1); text-align:left; color:black;">
            <h4>Clutch Performance</h4>
            <p>Quantifies performance under late-game pressure using situational models.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with preview_cols[3]:
    st.markdown(
        """
        <div style="padding:15px; border-radius:12px; background-color:#EBF5FB;
                    box-shadow:0px 4px 8px rgba(0,0,0,0.1); text-align:left; color:black;">
            <h4>Schedule Predictor</h4>
            <p>Forecasts future game outcomes and strength-of-schedule adjustments.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
