import streamlit as st

st.set_page_config(
    page_title = "Sports Prediction System",
    layout = "centered"
)

st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="stSidebarCollapsedControl"] {
        display: block;
    }
    [data-testid="stPageLink"] a span {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stPageLink"] p {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stPageLink"] a {
        padding-left: 0.5rem !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Sports Prediction System")
st.markdown("Machine Learning models that predict IndyCar Driver race Results, Liga MX Team Match Results Probabilities, UFC round-by-round fight outcomes and soon other sports")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/indycar_page.py", label="IndyCar Page", use_container_width=True)
    st.caption("""
    **IndyCar 2026 Season**
    - Pre & Post Qualifying race simulator  
    - Full field position predictions
    - Model Predicted results vs Actual results tab
    """)


with col2:
    st.page_link("pages/LigaMX_page.py", label="Liga MX Page", use_container_width=True)
    st.caption("""
    **Liga MX Clausura 2026**
    - Match result prediction (W/L/D)
    - Win, Loss, Draw probablilities
    -Model Predicted results vs Actual results tab
    """)

col_ufc, _ = st.columns(2)
with col_ufc:
    st.page_link("pages/UFC_page.py", label="UFC Page", use_container_width=True)
    st.caption("""
    **UFC Men's Divisions**
    - Each fighter's chance to win in every round or by decision
    - Tale of the tape for any matchup
    - Model vs actual results for every fight since 2025
    """)


st.divider()

st.markdown("#### Coming Soon")
col3, col4, col5, col7 = st.columns(4)
with col3:
    st.markdown("**Formula 1**")
    st.caption("Driver Race Position Prediction & Results tab")

with col4:
    st.markdown("**MLB**")
    st.caption("Match Result Prediction & Results tab")

with col5:
    st.markdown("**NBA**")
    st.caption("Match Result Prediction & Results tab")

with col7:
    st.markdown("**NFL**")
    st.caption("Match Result Prediction & Results tab")

st.divider()