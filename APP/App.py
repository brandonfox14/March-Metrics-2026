import streamlit as st
import os
import importlib.util
from PIL import Image

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="March Metrics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# SIDEBAR CONTENT
# -------------------------------------------------------------
logo_path = os.path.join("Assets", "Logos", "FullLogo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)
else:
    st.sidebar.warning("⚠️ Logo not found in Assets/Logos/FullLogo.png")

st.sidebar.title("March Metrics 2026")

st.sidebar.markdown(
    """
**March Metrics** combines **data science**, **machine learning**,  
and **basketball analytics** to give you a complete competitive edge.

Navigate below to explore **team models**, **player analysis**,  
and **predictive insights** for upcoming games.
"""
)
st.sidebar.divider()

# -------------------------------------------------------------
# PAGE DISCOVERY
# -------------------------------------------------------------
pages_dir = os.path.join(os.path.dirname(__file__), "Pages")

# List all Python page files
page_files = [
    f for f in sorted(os.listdir(pages_dir))
    if f.endswith(".py") and not f.startswith("_")
]

# Create user-friendly names
page_names = []
for file in page_files:
    clean = file.replace(".py", "").replace("_", " ")
    # Remove numeric prefix if present
    parts = clean.split(" ", 1)
    name = parts[1] if parts[0].isdigit() and len(parts) > 1 else clean
    page_names.append(name)

# Sidebar Navigation
selected_page = st.sidebar.radio("Navigation", page_names)

# -------------------------------------------------------------
# DYNAMIC PAGE IMPORT
# -------------------------------------------------------------
selected_file = page_files[page_names.index(selected_page)]
page_path = os.path.join(pages_dir, selected_file)

# Dynamically load and execute the selected page
spec = importlib.util.spec_from_file_location("page_module", page_path)
page_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(page_module)

# -------------------------------------------------------------
# SIDEBAR FOOTER
# -------------------------------------------------------------
st.sidebar.divider()
st.sidebar.info(
    "This app uses statistical modeling, regression analysis, and machine learning "
    "to evaluate teams, players, and game outcomes in college basketball."
)
