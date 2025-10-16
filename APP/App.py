import streamlit as st
import os

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="March Metrics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# SIDEBAR LOGO / INFO
# -------------------------------------------------------------
logo_path = os.path.join("Assets", "Logos", "FullLogo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)
else:
    st.sidebar.warning("⚠️ FullLogo.png not found in Assets/Logos/")

st.sidebar.title("🏀 March Metrics 2026")
st.sidebar.markdown("---")

# -------------------------------------------------------------
# DISCOVER PAGES AUTOMATICALLY
# -------------------------------------------------------------
pages_dir = os.path.join(os.path.dirname(__file__), "Pages")

# Create an ordered list of available page files
page_files = sorted(
    [f for f in os.listdir(pages_dir) if f.endswith(".py") and not f.startswith("_")]
)

# Map filenames to display names
pages = {}
for file in page_files:
    # Remove numeric prefix and underscores for clean labels
    label = file.replace("_", " ").replace(".py", "")
    label = label.split(" ", 1)[-1] if label[0].isdigit() else label
    pages[file] = label

# -------------------------------------------------------------
# SIDEBAR NAVIGATION
# -------------------------------------------------------------
selection = st.sidebar.radio("Navigate", list(pages.values()))

st.sidebar.markdown("---")

# -------------------------------------------------------------
# RENDER SELECTED PAGE
# -------------------------------------------------------------
selected_file = [k for k, v in pages.items() if v == selection][0]
selected_path = os.path.join(pages_dir, selected_file)

# Run the selected page
with open(selected_path, "r", encoding="utf-8") as f:
    code = f.read()
exec(code, globals())
