import streamlit as st
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="March Metrics",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR BRANDING ---
logo = Image.open("Assets/Logos/FullLogo.png")
st.sidebar.image(logo, use_container_width=True)
st.sidebar.markdown("### 🏀 March Metrics Dashboard")
st.sidebar.markdown("---")

# --- SIDEBAR INFO ---
st.sidebar.markdown("#### Navigation")
st.sidebar.markdown("""
Use the sidebar to switch between pages.
Streamlit automatically detects pages from the `/Pages` folder.
""")

st.sidebar.markdown("---")
st.sidebar.caption("© 2025 March Metrics | Internal Analytics Tool")

# --- MAIN PAGE DISPLAY ---
st.title("Welcome to March Metrics")
st.markdown("""
### Overview
This is the main entry point for your NCAA basketball analytics suite.  
Use the sidebar or page navigation tabs below to explore team metrics, player breakdowns, predictive analytics, and betting strategy tools.
""")

st.markdown("---")

# --- OPTIONAL QUICK LINKS ---
cols = st.columns(4)
with cols[0]:
    st.page_link("Pages/1 Main.py", label="Main Dashboard", icon="🏠")
with cols[1]:
    st.page_link("Pages/2 Team Breakdown.py", label="Team Breakdown", icon="📊")
with cols[2]:
    st.page_link("Pages/6 Todays Games.py", label="Today's Games", icon="📅")
with cols[3]:
    st.page_link("Pages/8 Betting.py", label="Betting (Private)", icon="💰")
