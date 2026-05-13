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
st.markdown("Machine Learning models that predict IndyCar Driver race Resutls, Liga MX Team Match Results Probabilities and soon other sports")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/indycar_page.py", label="IndyCar Page", use_container_width=True)
    st.caption("""
    **INDY 500 Month - MAY UPDATE**
    - Pre & Post Qualifying race simulator  
    - Full field position predictions
    - Model Predictedresults vs Actual results tab
    """)


with col2:
    st.page_link("pages/LigaMX_page.py", label="Liga MX Page", use_container_width=True)
    st.caption("""
    **Liga MX Clausura 2026**
    - Match result prediction (W/L/D)
    - Win, Loss, Draw probablilities
    -Model Predictedresults vs Actual results tab
    """)


st.divider()

st.markdown("#### Coming Soon")
col3, col4, col5, col6, col7 = st.columns(5)
with col3:
    st.markdown("**Formula 1**")
    st.caption("Driver Race Position Prediction & Results tab")

with col4:
    st.markdown("**MLB**")
    st.caption("Match Result Prediction & Results tab")

with col5:
    st.markdown("**NBA**")
    st.caption("Match Result Prediction & Results tab")

with col6:
    st.markdown("**UFC**")
    st.caption("Fight Result Prediction & Results tab")

with col7:
    st.markdown("**NFL**")
    st.caption("Match Result Prediction & Results tab")

st.divider()