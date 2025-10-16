# -*- coding: utf-8 -*-
# Home / Landing - EEG + NHIS Explorer

import os
import streamlit as st

st.set_page_config(
    page_title="EEG + NHIS Explorer",
    page_icon="🧭",
    layout="wide",
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def page_exists(path: str) -> bool:
    """Best-effort check for a page file under ./pages or root."""
    candidates = [path, os.path.join("pages", path), os.path.join(".", path)]
    return any(os.path.exists(p) for p in candidates)

def link_or_hint(label: str, page_path: str, icon: str = "→"):
    """Use st.page_link if available, else hint to use sidebar."""
    pl = getattr(st, "page_link", None)
    if pl and page_exists(page_path):
        pl(page=page_path, label=f"{icon} {label}")
    else:
        st.caption(f"{icon} Open via the **sidebar**: {label}")

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🧭 EEG + NHIS Explorer")
st.markdown(
    "Explore how **laboratory EEG** findings relate to **population-level sleep** patterns from the "
    "**National Health Interview Survey (NHIS)**. Built for learning, not diagnosis."
)
st.info(
    "This app is **educational** and **descriptive**. Data are de-identified. "
    "Comparisons are conceptual - datasets are not linked at the person level."
)

# ---------------------------------------------------------------------
# Datasets quick facts
# ---------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.markdown("### Lab dataset (OpenNeuro EEG)")
    st.markdown(
        "- Within-subject: **Normal Sleep (NS)** vs **Sleep Deprived (SD)**  \n"
        "- Resting EEG (eyes open/closed), **mood** (PANAS), **attention** (*PVT – Psychomotor Vigilance Test*), sleep questionnaires"
    )
with c2:
    st.markdown("### Survey dataset (NHIS)")
    st.markdown(
        "- U.S. household survey (2024)  \n"
        "- **Sleep hours**, restfulness, trouble sleeping, sleep-aid use  \n"
        "- **Demographics**: age, sex, education, race/ethnicity"
    )

st.divider()

# ---------------------------------------------------------------------
# Feature tiles
# ---------------------------------------------------------------------
st.markdown("## What you can do here")
tiles = st.columns(2)

with tiles[0]:
    st.markdown("#### 🧠 EEG Viewer")
    st.caption("Pick a participant, condition, and task. See EEG signals and a brain map with channel meanings.")
    link_or_hint("EEG Viewer", "1_🧠 EEG Viewer.py", icon="🧠")

    st.markdown("#### 📈 EEG Dashboard")
    st.caption("Interactive mood (PANAS), attention (*PVT – Psychomotor Vigilance Test*), and band-power views with plain-language explainers.")
    link_or_hint("EEG Dashboard", "2_📈 EEG Dashboard.py", icon="📈")

    st.markdown("#### ⚡ Reaction Time (PVT) Demo")
    st.caption("Try a simple reaction-time demo based on the Psychomotor Vigilance Test to feel how sleep loss can slow responses.")
    link_or_hint("Reaction Test", "reaction_test.py", icon="⚡")

with tiles[1]:
    st.markdown("#### 🗺️ NHIS Dashboard")
    st.caption("Explore U.S. sleep patterns by demographics: hours, restfulness, trouble sleeping.")
    link_or_hint("NHIS Dashboard", "nhis_dashboard.py", icon="🗺️")

    st.markdown("#### 🔗 Lab ↔ Survey Comparison")
    st.caption("Side-by-side views - e.g., alpha/theta vs self-reported sleep; mood/attention vs survey sleep.")
    link_or_hint("Lab vs Survey", "comparison_lab_vs_nhis.py", icon="🔗")

st.divider()

# ---------------------------------------------------------------------
# How we compare lab and survey data
# ---------------------------------------------------------------------
st.markdown("## How we compare lab and survey data")
with st.expander("Short rationale", expanded=False):
    st.markdown(
        "- **Different lenses**: EEG shows moment-to-moment brain activity and performance; "
        "NHIS shows how people **report** sleep and well-being across the U.S.  \n"
        "- **Complementary**: Lab results help explain mechanisms; survey patterns show **who** is most affected.  \n"
        "- **Examples**:  \n"
        "  • Eyes-closed **alpha** often rises; **theta** can rise when alertness drops - compare to NHIS **sleep hours** and **restfulness**.  \n"
        "  • **PANAS** (mood) and **PVT – Psychomotor Vigilance Test** (attention) under SD - compare to NHIS **trouble sleeping** and **daytime impacts**."
    )
st.caption("All findings are exploratory. We avoid causal claims and note key assumptions on each page.")

st.divider()

# ---------------------------------------------------------------------
# Data sources and reproducibility
# ---------------------------------------------------------------------
st.markdown("## Data sources and reproducibility")
st.markdown(
    "- **OpenNeuro EEG dataset**: Resting-state EEG for sleep deprivation "
    "(71 participants; eyes open/closed; mood and vigilance measures).  \n"
    "  - Dataset page: https://openneuro.org/datasets/ds004902/versions/1.0.8  \n"
    "  - Project repo: https://github.com/OpenNeuroDatasets/ds004902  \n"
    "- **NHIS**: 2024 public-use microdata, questionnaires, and documentation:  \n"
    "  - CDC NHIS hub: https://www.cdc.gov/nchs/nhis/index.html  \n"
    "  - 2024 documentation: https://www.cdc.gov/nchs/nhis/documentation/2024-nhis.html  \n"
    "- **Reproducibility**: processing steps are noted in each module; code paths and assumptions "
    "are documented in app text. The app is descriptive, not diagnostic."
)

st.divider()

# ---------------------------------------------------------------------
# Evidence behind the app (clickable, peer-reviewed) + relevance
# ---------------------------------------------------------------------
st.markdown("## 📚 Evidence behind the app")
with st.expander("Tap to see key studies that inform our choices", expanded=False):

    st.markdown("**EEG patterns**")
    st.markdown(
        "- Barry et al., 2007, *Clinical Neurophysiology*: eyes-closed vs eyes-open spectral differences  \n"
        "  https://www.sciencedirect.com/science/article/abs/pii/S1388245707004002  \n"
        "  *Relevance:* supports alpha increases with eyes closed used in EEG visuals.\n"
        "- Cajochen et al., 1995, *Sleep*: theta/alpha power rises across sustained wakefulness  \n"
        "  https://pubmed.ncbi.nlm.nih.gov/8746397/  \n"
        "  *Relevance:* informs NS vs SD comparisons for theta.\n"
        "- Berger, 1929 (historical): first human EEG and alpha suppression with eye opening  \n"
        "  Overview: https://pmc.ncbi.nlm.nih.gov/articles/PMC3740477/  \n"
        "  *Relevance:* context for alpha as a key EEG feature."
    )

    st.markdown("**Attention (PVT)**")
    st.markdown(
        "- Dinges & Powell, 1985, *Behavior Research Methods, Instruments, & Computers*: classic portable PVT  \n"
        "  https://link.springer.com/article/10.3758/BF03200977  \n"
        "  *Relevance:* underpins our PVT metrics and reaction-time demo.\n"
        "- Lim & Dinges, 2010, *Psychological Bulletin*: review/meta-analysis of neurobehavioral effects  \n"
        "  https://pmc.ncbi.nlm.nih.gov/articles/PMC3290659/  \n"
        "  *Relevance:* explains slower reaction times and lapses after sleep loss.\n"
        "- Basner & Dinges, 2011, *Sleep*: maximizing PVT sensitivity to sleep loss  \n"
        "  https://pubmed.ncbi.nlm.nih.gov/21532951/  \n"
        "  *Relevance:* supports selection of concise PVT summaries.\n"
        "- Basner, Mollicone & Dinges, 2011, *Acta Astronautica*: 3‑minute PVT‑B validity (free full text)  \n"
        "  https://pubmed.ncbi.nlm.nih.gov/22025811/  \n"
        "  *Relevance:* justifies brief PVT variants."
    )

    st.markdown("**Mood (PANAS)**")
    st.markdown(
        "- Watson, Clark & Tellegen, 1988, *Journal of Personality and Social Psychology*: PANAS development  \n"
        "  https://pubmed.ncbi.nlm.nih.gov/3397865/  \n"
        "  *Relevance:* validates Positive and Negative Affect scales used in the dashboard.\n"
        "- Pilcher & Huffcutt, 1996, *Sleep*: meta-analysis of sleep deprivation effects on performance and mood  \n"
        "  https://pubmed.ncbi.nlm.nih.gov/8776790/  \n"
        "  *Relevance:* supports interpreting PANAS differences between NS and SD."
    )

    st.markdown("**Sleep questionnaires and self-report**")
    st.markdown(
        "- Buysse et al., 1989, *Psychiatry Research*: Pittsburgh Sleep Quality Index (PSQI) validation  \n"
        "  https://pubmed.ncbi.nlm.nih.gov/2748771/  \n"
        "  *Relevance:* connects NHIS-style items to validated sleep-quality constructs.\n"
        "- Lauderdale et al., 2008, *Epidemiology*: self-reported vs actigraph-measured sleep duration  \n"
        "  https://journals.lww.com/epidem/Fulltext/2008/11000/Self_Reported_and_Measured_Sleep_Duration__How.15.aspx  \n"
        "  *Relevance:* clarifies why survey sleep can differ from objective measures."
    )

st.divider()

# ---------------------------------------------------------------------
# Design, accessibility, and transparency
# ---------------------------------------------------------------------
st.markdown("## Design, accessibility, and transparency")
colA, colB, colC = st.columns(3)
with colA:
    st.markdown("**Plain language**  \nTooltips, definitions, and 'How to read' sections accompany charts.")
with colB:
    st.markdown("**Color-blind friendly**  \nConsistent palette across EEG and NHIS modules.")
with colC:
    st.markdown("**Explainability**  \nEach module states why the visualization fits the question.")

st.caption(
    "Questions or suggestions? See each page's notes for assumptions, missingness, and data caveats."
)

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.divider()
st.caption(
    "EEG: OpenNeuro sleep-deprivation dataset • NHIS: 2024 public microdata • "
    "This app is non-diagnostic and for learning purposes only."
)
