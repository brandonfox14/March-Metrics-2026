import streamlit as st
from PIL import Image

# --- GLOBAL SETTINGS ---
st.set_page_config(
    page_title="March Metrics",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SIDEBAR BRANDING ---
logo = Image.open("Assets/Logos/FullLogo.png")
st.sidebar.image(logo, use_container_width=True)
st.sidebar.title("March Metrics")

st.sidebar.write("""
**A data-driven basketball analytics suite**  
powered by advanced **data science**, **machine learning**,  
and statistical modeling — giving you the ultimate edge  
in understanding every aspect of the game.
""")

st.sidebar.markdown("---")

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    [
        "Main",
        "Team Breakdown",
        "Team Comparison",
        "Clutch",
        "Schedule Predictor",
        "Today's Games",
        "Players Breakdown",
        "Betting"
    ]
)

# --- PAGE ROUTER ---
if page == "Main":
    from Pages.Main import main
    main()
elif page == "Team Breakdown":
    from Pages.TeamBreakdown import team_breakdown
    team_breakdown()
elif page == "Team Comparison":
    from Pages.TeamComparison import team_comparison
    team_comparison()
elif page == "Clutch":
    from Pages.Clutch import clutch
    clutch()
elif page == "Schedule Predictor":
    from Pages.SchedulePredictor import schedule_predictor
    schedule_predictor()
elif page == "Today's Games":
    from Pages.TodaysGames import todays_games
    todays_games()
elif page == "Players Breakdown":
    from Pages.PlayersBreakdown import players_breakdown
    players_breakdown()
elif page == "Betting":
    from Pages.Betting import betting
    betting()
