import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

Base_dir = Path(__file__).resolve().parent.parent
PREDICTIONS = Base_dir/"predictions"/"model_compare.xlsx"

# Load data
@st.cache_data
def load_predictions(filepath: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, sheet_name=sheet)
    df.columns = df.columns.str.strip()
    return df


# Helper functions
def error_color(val):
    if val in ("", "-"):
        return ""

    try:
        n = int(val)
        if abs(n) <= 2:
            return "color: green"
        elif abs(n) <= 5:
            return "color: yellow"
        return "color: red"
    except:
        return ""

def fmt_score(val):
    return f"{val:.2f}" if pd.notna(val) else "-"

def fmt_pos(val):
    return f"P{int(val)}" if pd.notna(val) else "-"

# UI

st.set_page_config(page_title="IndyCar Predictions Page", layout="centered")
st.title("Indycar 2026 - Prediction Results")
st.markdown("Pre and post Qualifying predictions of model & personal predictions based on models")

try:
    df_pre = load_predictions(str(PREDICTIONS), "Pre-Qualy Results")
    df_post = load_predictions(str(PREDICTIONS), "Post-Qualy Results")
except Exception as e:
    st.error(f"COuld not load predictions file 404: {e}")
    st.stop()

st.subheader("Filters")
col_phase, col_track = st.columns(2)

with col_phase:
    phase = st.radio("Phase:", ["Pre-Qualy", "Post-Qualy"], horizontal=True)

df = df_pre if phase == "Pre-Qualy" else df_post

with col_track:
    tracks = sorted(df["Track"].dropna().unique().tolist())
    selected_track = st.selectbox("Track:", tracks)

df_track = df[df["Track"] == selected_track].copy().sort_values("ModelRank").reset_index(drop=True)

if df_track.empty:
    st.warning("No predictions found for this track.")
    st.stop()


# Track metrics
has_results = df_track["ActualFinish"].notna().any()
track_type = df_track["TrackType"].iloc[0] if "TrackType" in df_track.columns else "-"
model_mae = df_track["ModelMAE"].iloc[0] if "ModelMAE" in df_track.columns and pd.notna(df_track["ModelMAE"].iloc[0]) else None

st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric("TrackType", track_type)
c2.metric("Phase", phase)
c3.metric("Results", "Available" if has_results else "Pending")
c4.metric("Model MAE", f"{model_mae:.2f} pos" if model_mae is not None else "-")

st.divider()

# Predictions table

st.subheader("Predictions")

model_col, my_col = st.columns(2)

with model_col:
    st.caption("Model Predictions")
    model_df = df_track[["ModelRank", "Driver", "RawScore", "Low", "High", "ActualFinish"]].copy()
    model_df = model_df.sort_values("ModelRank").reset_index(drop=True)
    model_df.columns = ["Position", "Driver", "Score", "Low", "High", "ActualFinish"]

    if has_results:
        model_df["Actual"] = model_df["ActualFinish"].apply(fmt_pos)
        model_df["Error"] = model_df.apply(
            lambda r: (f"+{int(r['ActualFinish'] - r['Position'])}"
                       if r['ActualFinish'] - r['Position'] > 0
                       else str(int(r['ActualFinish'] - r['Position'])))
            if pd.notna(r['ActualFinish']) else "-", axis=1
        )
    
    model_df = model_df.drop(columns=["ActualFinish"])

    styled_model = model_df.style.format({"Score": "{:.2f}", "Low": "{:.2f}", "High": "{:.2f}"})
    if has_results:
        styled_model = styled_model.applymap(error_color, subset=["Error"])

    st.dataframe(styled_model, hide_index=True, use_container_width=True)

with my_col:
    st.caption("My Predictions")
    my_df = df_track[["MyRank", "Driver", "MyRawScore"]].copy()
    my_df = my_df.sort_values("MyRank").reset_index(drop=True)
    my_df.columns = ["Position", "Driver", "Score"]

    st.dataframe(
        my_df.style.format({"Score": "{:.2f}"}),
        hide_index=True,
        use_container_width=True
        )


st.divider()

# Model vs Mine Graph
st.subheader("Model Position Prediction vs My Position Prediction")

chart_df = df_track[["Driver", "ModelRank", "MyRank"]].dropna().copy()
chart_df["ModelRank"] = chart_df["ModelRank"].astype(int)
chart_df["MyRank"] = chart_df["MyRank"].astype(int)
chart_df["Diff"] = (chart_df["ModelRank"]- chart_df["MyRank"]).abs()
chart_df["Agreement"] = chart_df["Diff"].apply(
    lambda x: "Agree" if x <= 2 else ("Perhaps" if x <= 5 else "Disagree"))

graph = alt.Chart(chart_df).mark_circle(size=90, opacity=0.85).encode(
    x=alt.X("ModelRank:Q", title="Model Prediction", scale=alt.Scale(domain=[0,26])),
    y=alt.Y("MyRank:Q", title="My Prediction", scale=alt.Scale(domain=[0,26])),
    color=alt.Color("Agreement:N", scale=alt.Scale(
        domain=["Agree", "Perhaps", "Disagree"],
        range= ["#3B6D11", "#F5DA27", "#A32D2D"]),
                    legend=alt.Legend(title="Pos Agreement")),
    tooltip=["Driver:N", "ModelRank:Q", "MyRank:Q", "Diff:Q"]
).properties(height=380)

diag_df = pd.DataFrame({"x": [1,25], "y": [1,25]})
diag = alt.Chart(diag_df).mark_line(
    color="gray", strokeDash=[4,4], opacity=.45).encode(x="x:Q", y="y:Q")

st.altair_chart(graph + diag, use_container_width=True)


# Actual vs Predicted (After race is complete)
if has_results:
    st.divider()
    st.subheader("Actual vs predicted finish")
    result_df = df_track[["Driver", "ModelRank", "ActualFinish"]].dropna().copy()
    result_df["ModelRank"] = result_df["ModelRank"].astype(int)
    result_df["ActualFinish"] = result_df["ActualFinish"].astype(int)
    result_df["Error"] = (result_df["ActualFinish"] - result_df["ModelRank"]).abs()
    result_df = result_df.sort_values("ActualFinish").reset_index(drop=True)

    bar = alt.Chart(result_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Driver:N", sort=alt.EncodingSortField(field="ActualFinish", order="ascending"), title=""),
        y=alt.Y("Error:Q", title="Position Error"),
        color=alt.condition(
            alt.datum.Error <= 2,
            alt.value("#3B6D11"),
            alt.condition(
                alt.datum.Error <= 5,
                alt.value("#F5DA27"),
                alt.value("#A32D2D")
                )
            ),
        tooltip=["Driver:N",
                 alt.Tooltip("ModelRank:Q", title="Predicted"),
                 alt.Tooltip("ActualFinish:Q", title="Actual"),
                 alt.Tooltip("Error:Q", title="Error")]).properties(height=380)

    st.altair_chart(bar, use_container_width=True)

    computed_mae = result_df["Error"].mean()
    best = result_df.loc[result_df["Error"].idxmin(), "Driver"]
    worst = result_df.loc[result_df["Error"].idxmax(), "Driver"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Actual MAE", f"{computed_mae:.2f} positions")
    c2.metric("Best prediction", best)
    c3.metric("Worst prediction", worst)

    