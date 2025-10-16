import streamlit as st
import pandas as pd

# --- PAGE CONFIG ---
st.set_page_config(page_title="March Metrics", layout="wide")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("Data/26_March_Madness_Databook/All_Stats-THE_TABLE.csv", encoding="latin1")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# --- HEADER WITH LOGO + TITLE ---
col1, col2 = st.columns([3, 1])

with col1:
    st.title("🏀 March Metrics")

with col2:
    st.image("Assets/Logos/FullLogo.png", use_container_width=True)

st.markdown("---")

# --- TOP 10 TEAMS BY STATISTICAL STRENGTH ---
st.subheader("🔥 Top 10 Teams by Statistical Strength")

# Ensure STAT_STREN is numeric
df["STAT_STREN"] = pd.to_numeric(df["STAT_STREN"], errors="coerce")

# Sort and filter columns
top10 = (
    df[["Teams", "Coach Name", "Conference", "Wins", "Losses", "STAT_STREN"]]
    .sort_values(by="STAT_STREN", ascending=True)  # lower = better
    .head(10)
    .reset_index(drop=True)
)
top10.index = top10.index + 1  # rank 1-10

# Show styled dataframe
st.dataframe(
    top10.style.format({
        "STAT_STREN": "{:.3f}",
        "Wins": "{:.0f}",
        "Losses": "{:.0f}"
    }).background_gradient(subset=["STAT_STREN"], cmap="Greens_r"),
    use_container_width=True
)

# --- SITE NAVIGATION LINKS ---
st.markdown("---")
st.subheader("📍 Quick Navigation")

nav_cols = st.columns(4)

with nav_cols[0]:
    st.markdown(
        """
        <a href="/Team_Breakdown" target="_self" style="text-decoration:none;">
            <div style="padding:15px; border-radius:12px; background-color:#E8F8F5;
                        box-shadow:0px 4px 8px rgba(0,0,0,0.1); text-align:center; color:black;">
                <h4>Team Breakdown</h4>
                <p>Dive into team analytics.</p>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )

with nav_cols[1]:
    st.markdown(
        """
        <a href="/Team_Comparison" target="_self" style="text-decoration:none;">
            <div style="padding:15px; border-radius:12px; background-color:#FEF9E7;
                        box-shadow:0px 4px 8px rgba(0,0,0,0.1); text-align:center; color:black;">
                <h4>Team Comparison</h4>
                <p>Compare multiple teams head-to-head.</p>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )

with nav_cols[2]:
    st.markdown(
        """
        <a href="/Clutch" target="_self" style="text-decoration:none;">
            <div style="padding:15px; border-radius:12px; background-color:#FDEDEC;
                        box-shadow:0px 4px 8px rgba(0,0,0,0.1); text-align:center; color:black;">
                <h4>Clutch Metrics</h4>
                <p>Late-game impact and win probability.</p>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )

with nav_cols[3]:
    st.markdown(
        """
        <a href="/Schedule_Predictor" target="_self" style="text-decoration:none;">
            <div style="padding:15px; border-radius:12px; background-color:#EBF5FB;
                        box-shadow:0px 4px 8px rgba(0,0,0,0.1); text-align:center; color:black;">
                <h4>Schedule Predictor</h4>
                <p>Forecast game outcomes and scenarios.</p>
            </div>
        </a>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# --- CONFIDENTIALITY NOTE ---
st.info(
    """
    **Note:** This demo displays limited data to preserve business confidentiality.  
    Full model outputs and betting strategies are private to March Metrics.
    """
)
