# App.py
import streamlit as st
from PIL import Image
import os

# -------------------------------------------------------------
# GLOBAL PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="March Metrics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# SIDEBAR CONFIGURATION
# -------------------------------------------------------------
# Load logo safely
logo_path = "Assets/Logos/FullLogo.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)
else:
    st.sidebar.warning("⚠️ FullLogo.png not found in Assets/Logos/")

st.sidebar.title("March Metrics 2026")

st.sidebar.markdown(
    """
**March Metrics** is a college basketball analytics platform combining  
**machine learning**, **data science**, and **basketball intelligence**  
to deliver competitive insights and predictive power.
"""
)

st.sidebar.divider()

# -------------------------------------------------------------
# DYNAMICALLY ADD PAGE LINKS
# -------------------------------------------------------------
st.sidebar.header("📂 Navigation")

# Manually list pages (Streamlit automatically detects them too, but this gives full control)
pages = {
    "Main Dashboard": "Pages/1_Main.py",
    "Team Breakdown": "Pages/2_Team_Breakdown.py",
    "Team Comparison": "Pages/3_Team_Comparison.py",
    "Clutch": "Pages/4_Clutch.py",
    "Schedule Predictor": "Pages/5_Schedule_Predictor.py",
    "Today's Games": "Pages/6_Todays_Games.py",
    "Players Breakdown": "Pages/7_Players_Breakdown.py",
    "Betting": "Pages/8_Betting.py",
}

for name, path in pages.items():
    st.sidebar.page_link(path, label=name)

st.sidebar.divider()
st.sidebar.info("Use the sidebar to switch between analytic modules.")

# -------------------------------------------------------------
# MAIN PAGE CONTENT
# -------------------------------------------------------------
st.title("March Metrics: Data-Driven Basketball Analytics")

st.write(
    """
Welcome to **March Metrics**, a **data science–powered platform** for analyzing and 
predicting college basketball performance.

This application integrates:
- **Statistical modeling** and regression analysis  
- **Machine learning** for predictive outcomes  
- **Context-adjusted efficiency metrics** (e.g., SOS, SM+, Coach Value)  
- **Game-level and player-level analysis**  
- **Betting and projection models** leveraging historical trends

Our goal is to provide the most **comprehensive, data-driven view** of how teams 
and players perform—both in-season and during March Madness.
"""
)

st.markdown("---")

st.subheader("Core Capabilities")
cols = st.columns(3)

with cols[0]:
    st.markdown("""
    #### Team Analytics  
    Explore team-level metrics, performance distributions, and situational tendencies.
    """)

with cols[1]:
    st.markdown("""
    #### Predictive Modeling  
    Apply machine learning to forecast outcomes and simulate tournament matchups.
    """)

with cols[2]:
    st.markdown("""
    #### Betting Insights  
    Analyze line value, edge probabilities, and optimized betting strategies.
    """)

st.markdown("---")

st.subheader("Data Overview")
st.write("""
All data is sourced from the **26_March_Madness_Databook**, containing:
- Team, player, and coach performance datasets  
- Strength-of-schedule indexing and historical models  
- Clutch-time and contextual analytics  
- Predictive frameworks for win probability and statistical strength  
""")

st.info(
    "Note: This portfolio version omits full proprietary data and ML models for confidentiality."
)
