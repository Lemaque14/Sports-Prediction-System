import streamlit as st

st.set_page_config(
    page_title = "Sports Prediction System",
    layout = "centered"
)

st.title("Sports Prediction System")
st.markdown("Machine Learning models that predict IndyCar Driver race Resutls, Liga MX Team Match Results Probabilities and soon other sports")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("IndyCar")
    st.markdown("""
    **Indycar Series 2026**
    - Pre & Post Qualifying race simulator  
    - Full field position predictions
    - Model Predictedresults vs Actual results tab
    """)
    st.page_link("pages/indycar_page.py", label="IndyCar Page", use_container_width=True)

with col2:
    st.subheader("Liga MX")
    st.markdown("""
    **Liga MX Clausura 2026**
    - Match result prediction (W/L/D)
    - Win, Loss, Draw probablilities
    -Model Predictedresults vs Actual results tab coming soon...
    """)
    st.page_link("pages/LigaMX_page.py", label="Liga MX Page", use_container_width=True)

st.divider()

st.markdown("#### Coming Soon")
col3, col4, col5, col6 = st.columns(4)
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
    st.markdown("**NFL**")
    st.caption("Match Result Prediction & Results tab") 

st.divider()