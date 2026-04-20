import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Liga MX Prediction Page", layout="centered")
st.title("Liga MX Clausura 2026")

Base_dir = Path(__file__).resolve().parent.parent / "LigaMX"

# Paths to models, datset and model prediction results
MODEL = Base_dir/"models"/"LigaMX_model_v1"
DATASET = Base_dir/"datasets"/"LigaMX_dataset_v2.csv"
PREDICTIONS = Base_dir/"predictions"/"model_compareLMX.xlsx"

# Load data
@st.cache_resource
def load_requested_model(model: str):
    return load_model(model)

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_data
def load_stats(path: str) -> dict:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df_sorted = df.sort_values("Date")

    team_stats = df_sorted.groupby("TeamID").last().to_dict("index")
    h2h_stats = df_sorted.groupby(["TeamID", "OpponentID"]).last().to_dict("index")

    return {"team": team_stats, "h2h": h2h_stats}

@st.cache_data
def load_maps(path: str) -> dict:
    df = pd.read_csv(path)
    team_map = dict(zip(df["Equipo"], df["TeamID"]))
    ref_map = dict(zip(df["Referee"].dropna(), df["RefereeID"].dropna()))
    torneo_map = dict(zip(df["Torneo"], df["TorneoID"]))
    temporada_map = dict(zip(df["Temporada"], df["TemporadaID"]))

    return{
        "team" : team_map,
        "ref": ref_map,
        "torneo" : torneo_map,
        "temporada" : temporada_map
    }

@st.cache_data
def load_results(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Results")
    df.columns = df.columns.str.strip()
    return df

# Helper function to create feature row
def populate_feature_row(team_id, opponent_id, venue_id, ref_id, round_id, torneo_id, temp_id, time_id, day_id, stats: dict) -> dict:
    team_row = stats["team"].get(team_id, {})
    opponent_row = stats["team"].get(opponent_id, {})
    h2h_row = stats["h2h"].get((team_id,opponent_id), {})

    return {
        "VenueID"           : venue_id,
        "OpponentID"        : opponent_id,
        "TeamID"            : team_id,
        "RefereeID"         : ref_id,
        "RoundID"           : round_id,
        "TemporadaID"       : 12,
        "TorneoID"          : torneo_id,
        "TimeID"            : time_id,
        "DayID"             : day_id,
        "TeamElo"           : team_row.get("TeamElo", 1500.0),
        "OponentElo"        : opponent_row.get("TeamElo", 1500.0),
        "EloDiff"           : team_row.get("TeamElo", 1500) - opponent_row.get("TeamElo", 1500),
        "TeamForm5"         : team_row.get("TeamForm5", 0),
        "TeamWinRate5"      : team_row.get("TeamWinRate5", 0),
        "TeamSeasonPts"     : team_row.get("TeamSeasonPts", 0),
        "TeamHomeForm5"     : team_row.get("TeamHomeForm5", 0),
        "TeamAwayForm5"     : team_row.get("TeamAwayForm5", 0),
        "OpponentForm5"     : opponent_row.get("TeamForm5", 0),
        "OpponentWinRate5"  : opponent_row.get("TeamWinRate5", 0),
        "OpponentSeasonPts" : opponent_row.get("TeamSeasonPts", 0),
        "OpponentHomeForm5" : opponent_row.get("TeamHomeForm5", 0),
        "OpponentAwayForm5" : opponent_row.get("TeamAwayForm5", 0),
        "H2HWinRate"        : h2h_row.get("H2HWinRate", 0),
        "H2HLast5"          : h2h_row.get("H2HLast5", 0),
        "FormDiff"          : team_row.get("TeamForm5", 0) - opponent_row.get("TeamForm5", 0),
        "SeasonPtsDiff"     : team_row.get("TeamSeasonPts", 0) - opponent_row.get("TeamSeasonPts", 0),
    }


try:
    model = load_requested_model(str(MODEL))
    df_data = load_data(str(DATASET))
    stats = load_stats(str(DATASET))
    maps = load_maps(str(DATASET))
except Exception as e:
    st.error(f"Could not load model or dataset: {e}")
    st.stop()


TEAMS = sorted(df_data["Equipo"].dropna().unique().tolist())
REFS = sorted(df_data["Referee"].dropna().unique().tolist())
DAY_MAP = {"Mon":0, "Tue":1, "Wed":2, "Thu":3, "Fri":4, "Sat":5, "Sun":6}
RESULT_MAP = {
    1  : "Win",
    0  : "Loss",
    2  : "Draw"
}
JORNADA_MAP = {  
    "Cuartos"   : 0,
    "Final"     : 1,
    "Jornada 1" : 2,
    "Jornada 10": 3,
    "Jornada 11": 4,
    "Jornada 12": 5,
    "Jornada 13": 6,
    "Jornada 14": 7,
    "Jornada 15": 8,
    "Jornada 16": 9,
    "Jornada 17": 10,
    "Jornada 18": 11,
    "Jornada 2" : 12,
    "Jornada 3" : 13,
    "Jornada 4" : 14,
    "Jornada 5" : 15,
    "Jornada 6" : 16,
    "Jornada 7" : 17,
    "Jornada 8" : 18,
    "Jornada 9" : 19,
    "Repechaje" : 20,
    "Semis"     : 21,
}

TIME_OPTIONS = {
    "12:00 PM": 12,
    "1:00 PM" : 13,
    "2:00PM"  : 14,
    "3:00 PM" : 15,
    "4:00 PM" : 16,
    "5:00 PM" : 17,
    "6:00 PM" : 18,
    "7:00 PM" : 19,
    "8:00 PM" : 20,
    "9:00 PM" : 21,
    "10:00 PM": 22,
}

# UI
st.page_link("sportssystem_homepage.py", label="Back to Homepage")
tab_sim, tab_result = st.tabs(["Macth Simulator", "Results"])

# Tab 1 - Macth Simulator
with tab_sim:
    st.subheader("Matches")
    st.caption("Liga MX Clausura 2026")


    col1, col2 = st.columns(2)

    with col1:
        team = st.selectbox("Team", TEAMS, key="team_selected")
    with col2:
        opponent = st.selectbox("Opponent", [t for t in TEAMS if t != team], key="opp_selected")

    col3, col4 = st.columns(2)
    with col3:
        venue_label = st.radio("Venue", ["Home", "Away"], horizontal=True)
        venue_id = 1 if venue_label == "Home" else 0
    with col4:
        ref = st.selectbox("Referee", REFS, index=None, placeholder="Type to search..", key="ref_selected")
    
    col5, col6, col7 = st.columns(3)
    with col5:
        jornada_label = st.selectbox("Jornada", list(JORNADA_MAP.keys()),index=2, key="jornada_selected")
        round_id = JORNADA_MAP[jornada_label]
    with col6:
        day_label = st.selectbox("Day", list(DAY_MAP.keys()), index=5, key="day_selected")
        day_id = DAY_MAP[day_label]
    with col7:
        time_label = st.selectbox("Time", list(TIME_OPTIONS.keys()), index=7, key="time_selected")
        time_id = TIME_OPTIONS[time_label]

    if st.button("Predict Result", use_container_width=True, key="ligamx_predict"):
        if team == opponent:
            st.warning("Team and opponent cannot be the same")
        else:
            try:
                team_id = maps["team"].get(team,0)
                opponent_id = maps["team"].get(opponent,0)
                ref_id = maps["ref"].get(ref, 0)
                torneo_id = maps["torneo"].get("Clausura", 0)
                temporada_id = maps["temporada"].get(2026, 0)
                time_id = time_id
                day_id = day_id
                round_id = round_id

                features = populate_feature_row(team_id, opponent_id, venue_id, ref_id, round_id, torneo_id, temporada_id, time_id, day_id, stats)

                input_df = pd.DataFrame([features])
                result = predict_model(model, data=input_df, raw_score=True, verbose=False)

                w_prob = float(result["prediction_score_1"].iloc[0])
                l_prob = float(result["prediction_score_0"].iloc[0])
                d_prob = float(result["prediction_score_2"].iloc[0])
                predicted = RESULT_MAP[result["prediction_label"].iloc[0]]

                st.divider()
                st.subheader(f"{team} vs {opponent}")

                outcome_color = {"Win":"green", "Loss":"red", "Draw":"orange"}
                st.markdown(f"### Predicted result: {team} :{outcome_color[predicted]}[**{predicted}**]")

                st.divider()

                col_w, col_d, col_l = st.columns(3)
                with col_w:
                    st.metric("Win", f"{w_prob:.2%}")
                    st.progress(float(w_prob))
                with col_d:
                    st.metric("Draw", f"{d_prob:.2%}")
                    st.progress(float(d_prob))
                with col_l:
                    st.metric("Loss", f"{l_prob:.2%}")
                    st.progress(float(l_prob))

                st.divider()

            except Exception as e:
                st.error(f"Prediction error: {e}")
with tab_result:
    try:
        df_results = load_results(str(PREDICTIONS))
    except Exception as e:
        st.error(f"Could not load results file: {e}")
        st.stop()
 
    df_results["Correct"] = df_results["Predicted Result"] == df_results["Actual Result"]
 
    jornadas     = sorted(df_results["Jornada"].unique().tolist())
    selected_jornada = st.selectbox(
        "Filter by Jornada",
        ["All"] + [str(j) for j in jornadas],
        key="results_jornada"
    )
 
    df_view = df_results.copy() if selected_jornada == "All" else \
              df_results[df_results["Jornada"] == int(selected_jornada)].copy()
 
    st.divider()
 
    total       = len(df_results)
    correct     = df_results["Correct"].sum()
    accuracy    = correct / total if total > 0 else 0
 
    # outcome accuracy
    c1, c2, c3, c4 = st.columns(4)

    wins   = df_results[df_results["Actual Result"].str.contains("Win",  na=False)]
    losses = df_results[df_results["Actual Result"].str.contains("Loss", na=False)]
    draws  = df_results[df_results["Actual Result"].str.contains("Draw", na=False)]

    acc_win  = wins[wins["Correct"]].shape[0]   / len(wins)  if len(wins)  > 0 else 0
    acc_loss = losses[losses["Correct"]].shape[0] / len(losses) if len(losses) > 0 else 0
    acc_draw = draws[draws["Correct"]].shape[0]  / len(draws) if len(draws) > 0 else 0

    c1.metric("Overall", f"{accuracy:.1%}", f"{correct}/{total}")
    c2.metric("Win",     f"{acc_win:.1%}",  f"{len(wins)} matches")
    c3.metric("Loss",    f"{acc_loss:.1%}", f"{len(losses)} matches")
    c4.metric("Draw",    f"{acc_draw:.1%}", f"{len(draws)} matches")
 
    st.divider()
 
    st.subheader("Accuracy per jornada")
    jornada_acc = (df_results.groupby("Jornada")
                   .apply(lambda x: pd.Series({
                       "Correct"  : x["Correct"].sum(),
                       "Total"    : len(x),
                       "Accuracy" : x["Correct"].mean()
                   })).reset_index())
 
    bar = alt.Chart(jornada_acc).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("Jornada:O", title="Jornada"),
        y=alt.Y("Accuracy:Q", title="Accuracy", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.condition(
            alt.datum.Accuracy >= 0.5,
            alt.value("#3B6D11"),
            alt.value("#A32D2D")
        ),
        tooltip=[
            alt.Tooltip("Jornada:O"),
            alt.Tooltip("Correct:Q", title="Correct"),
            alt.Tooltip("Total:Q",   title="Total"),
            alt.Tooltip("Accuracy:Q",title="Accuracy", format=".1%")
        ]
    ).properties(height=250)
 
    # Average line
    avg_line = alt.Chart(pd.DataFrame({"avg": [accuracy]})).mark_rule(
        color="gray", strokeDash=[4, 4], opacity=0.6
    ).encode(y="avg:Q")
 
    st.altair_chart(bar + avg_line, use_container_width=True)
    st.caption(f"Dashed line = season average ({accuracy:.1%})")
 
    st.divider()
 
    st.subheader(f"Predictions — {'All Jornadas' if selected_jornada == 'All' else f'Jornada {selected_jornada}'}")
 
    def color_correct(val):
        if val == True:
            return "color: green; font-weight: 500"
        return "color: red; font-weight: 500"
 
    def color_result(val):
        if "Win" in str(val):
            return "color: green"
        elif "Loss" in str(val):
            return "color: red"
        elif "Draw" in str(val):
            return "color: orange"
        return ""
 
    display_df = df_view[["Jornada", "Team", "Opponent", "Predicted Result", "Actual Result", "Correct"]].copy()
    display_df["Correct"] = display_df["Correct"].map({True: "Yes", False: "No"})
 
    styled = (display_df.style
              .applymap(color_result, subset=["Predicted Result", "Actual Result"])
              .applymap(lambda v: "color: green; font-weight:500" if v == "Yes" else "color: red; font-weight:500",
                        subset=["Correct"]))
 
    st.dataframe(styled, hide_index=True, use_container_width=True)
 
    if selected_jornada == "All":
        st.divider()
        st.subheader("Summary by jornada")
        summary = (df_results.groupby("Jornada")
                   .apply(lambda x: pd.Series({
                       "Matches" : len(x),
                       "Correct" : int(x["Correct"].sum()),
                       "Accuracy": f"{x['Correct'].mean():.1%}"
                   })).reset_index())
        st.dataframe(summary, hide_index=True, use_container_width=True)
    
