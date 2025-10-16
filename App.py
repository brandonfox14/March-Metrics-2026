import streamlit as st
from PIL import Image

# --- GLOBAL SETTINGS ---
st.set_page_config(
    page_title="March Metrics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR / HEADER ---
logo = Image.open("assets/logos/FullLogo.png")
st.sidebar.image(logo, use_container_width=True)
st.sidebar.title("March Metrics")

st.sidebar.write("""
**A data-driven basketball analytics suite**
built with advanced **data science**, **machine learning**,  
and statistical modeling — giving you the ultimate edge  
in understanding every angle of the game.
""")

st.sidebar.markdown("---")
st.sidebar.success("Use the sidebar to navigate between pages.")

# --- MAIN LANDING VIEW ---
st.title("🏀 March Metrics Dashboard")
st.write("""
Welcome to **March Metrics**, an advanced basketball intelligence platform.  
Here you can:
- Explore team breakdowns and efficiency trends  
- Compare programs with advanced data metrics  
- Analyze clutch performance and predictive models  
- Run schedule outcome predictions and betting edges  
- Track real-time player and game updates
""")

st.markdown("---")
st.subheader("How it works")

st.write("""
Each page in the sidebar gives you analytical access to different datasets and models.  
All data pipelines are powered by `pandas`, `scikit-learn`, and `xgboost`, integrating:
- Machine learning models for prediction  
- Regression-based value scoring  
- Context-adjusted performance metrics  
- Automated feature engineering and visualization
""")
