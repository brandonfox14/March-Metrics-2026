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
# Load company logo
try:
    logo = Image.open("Assets/Logos/FullLogo.png")
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("⚠️ FullLogo.png not found in Assets/Logos/")

st.sidebar.title("March Metrics 2026")

st.sidebar.markdown(
    """
**March Metrics** is a college basketball analytics platform built to  
combine **machine learning**, **data science**, and **basketball intelligence**  
for competitive insight and decision support.
    
Navigate the pages below to explore advanced models, predictive analytics,  
and data-driven insights across teams, players, and betting markets.
"""
)

st.sidebar.divider()
st.sidebar.success("Use the sidebar below to access each section of the app.")
st.sidebar.divider()

# -------------------------------------------------------------
# MANUAL NAVIGATION LINKS
# -------------------------------------------------------------
st.sidebar.markdown("### Navigation")

# Create a manual link to the 1_Main.py page
# (Streamlit pages can be opened via URL query string ?page=<page_name>)
main_page_path = os.path.join("Pages", "1_Main.py")

st.sidebar.page_link(main_page_path, label="🏠 Main Page")
st.sidebar.page_link("Pages/2_Team_Breakdown.py", label="📊 Team Breakdown")
st.sidebar.page_link("Pages/3_Team_Comparison.py", label="⚖️ Team Comparison")
st.sidebar.page_link("Pages/4_Clutch.py", label="🔥 Clutch")
st.sidebar.page_link("Pages/5_Schedule_Predictor.py", label="📅 Schedule Predictor")
st.sidebar.page_link("Pages/6_Todays_Games.py", label="🎯 Today's Games")
st.sidebar.page_link("Pages/7_Players_Breakdown.py", label="👤 Players Breakdown")
st.sidebar.page_link("Pages/8_Betting.py", label="💰 Betting")

# -------------------------------------------------------------
# MAIN PAGE CONTENT
# -------------------------------------------------------------
st.title("🏀 March Metrics: Data-Driven Basketball Analytics")

st.write(
    """
Welcome to **March Metrics**, a data science–driven platform for analyzing and 
predicting college basketball performance.

This application integrates:
- **Statistical modeling** and regression analysis  
- **Machine learning** for predictive outcomes  
- **Context-adjusted efficiency metrics** (e.g., SOS, SM+, Coach Value)  
- **Game-level and player-level breakdowns**  
- **Betting edge analysis** leveraging historical performance

The goal: provide an **unmatched analytical view** of how teams and players perform, 
both in-season and under tournament conditions.
"""
)

st.markdown("---")
st.subheader("🏆 Core Capabilities")

cols = st.columns(3)

with cols[0]:
    st.markdown("""
    #### 📊 Team Analytics  
    Explore team-level performance using advanced statistical metrics,  
    contextualized by opponent strength and location effects.
    """)

with cols[1]:
    st.markdown("""
    #### 🔍 Predictive Modeling  
    Apply regression and ML-based models to forecast outcomes and simulate  
    tournament matchups or future schedules.
    """)

with cols[2]:
    st.markdown("""
    #### 💰 Betting Analysis  
    Evaluate line value, expected outcomes, and potential betting edges  
    powered by context-driven model probabilities.
    """)

st.markdown("---")

st.subheader("📘 Data Overview")
st.write("""
The data powering this app is sourced from the **26_March_Madness_Databook**,  
which includes:
- Team, player, and coach performance datasets  
- Strength of schedule indexing and historical value tables  
- Clutch-time and situational metrics  
- Predictive models for statistical strength and win probability

All files are located under `Data/26_March_Madness_Databook/`.
""")

st.markdown("---")

st.info(
    "This version is a **portfolio-safe example**. Full datasets and proprietary modeling details are excluded for confidentiality."
)
