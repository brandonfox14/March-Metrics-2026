# App.py
import streamlit as st
from PIL import Image

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
# Load logo
try:
    logo = Image.open("Assets/Logos/FullLogo.png")
    st.sidebar.image(logo, use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("⚠️ FullLogo.png not found in Assets/Logos/")

st.sidebar.title("March Metrics 2026")

st.sidebar.markdown(
    """
**March Metrics** combines **machine learning**, **data science**,  
and **basketball intelligence** for unmatched analytical insight.  

Navigate the app to explore predictive models, team analytics,  
and betting optimization strategies.
"""
)

st.sidebar.divider()
st.sidebar.subheader("📂 Pages")

# -------------------------------------------------------------
# MANUAL PAGE LINKS
# -------------------------------------------------------------
# Use relative paths to link to your Streamlit pages
st.sidebar.page_link("App.py", label="🏠 Home / Overview")
st.sidebar.page_link("Pages/1 Main.py", label="📈 Main Dashboard")
st.sidebar.page_link("Pages/2 Team Breakdown.py", label="🏀 Team Breakdown")
st.sidebar.page_link("Pages/3 Team Comparison.py", label="⚖️ Team Comparison")
st.sidebar.page_link("Pages/4 Clutch.py", label="⏱️ Clutch Analysis")
st.sidebar.page_link("Pages/5 Schedule_Predictor.py", label="🗓️ Schedule Predictor")
st.sidebar.page_link("Pages/6 Todays Games.py", label="📅 Today's Games")
st.sidebar.page_link("Pages/7 Players Breakdown.py", label="👥 Player Breakdown")
st.sidebar.page_link("Pages/8 Betting.py", label="💰 Betting Insights")

st.sidebar.divider()
st.sidebar.info("Select a page above to explore March Metrics analytics.")

# -------------------------------------------------------------
# MAIN PAGE CONTENT
# -------------------------------------------------------------
st.title("🏀 March Metrics: Data-Driven Basketball Analytics")

st.write(
    """
Welcome to **March Metrics**, a cutting-edge platform for analyzing and 
predicting college basketball performance.

This application integrates:
- **Statistical modeling** and **regression analysis**
- **Machine learning** for predictive outcomes
- **Context-adjusted metrics** such as SOS, SM+, and Coach Value
- **Game-level and player-level analytics**
- **Betting edge detection** using historical and model-based probabilities

Our mission: to deliver **the most complete basketball analytics suite** 
for decision-makers, coaches, and predictive strategists.
"""
)

st.markdown("---")
st.subheader("🏆 Core Capabilities")

cols = st.columns(3)

with cols[0]:
    st.markdown("""
    #### 📊 Team Analytics  
    Deep-dive into advanced team metrics — adjusted for schedule,  
    opponent strength, and location.
    """)

with cols[1]:
    st.markdown("""
    #### 🔍 Predictive Modeling  
    Apply regression and machine learning models to forecast matchups,  
    rankings, and tournament outcomes.
    """)

with cols[2]:
    st.markdown("""
    #### 💰 Betting Optimization  
    Evaluate betting edges and risk-adjusted unit strategies  
    powered by probability modeling.
    """)

st.markdown("---")

st.subheader("📘 Data Overview")
st.write("""
Data is sourced from the **26_March_Madness_Databook**, including:
- Team, player, and coach performance data  
- Strength of schedule and context indexing  
- Clutch and efficiency metrics  
- Predictive modeling datasets

All located in `Data/26_March_Madness_Databook/`.
""")

st.markdown("---")

st.info(
    "This portfolio-safe version includes mock data and structures — full proprietary models are excluded for confidentiality."
)

