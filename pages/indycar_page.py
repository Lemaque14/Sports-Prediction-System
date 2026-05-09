import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
from pathlib import Path
from pycaret.regression import load_model, predict_model


st.set_page_config(page_title="IndyCar Prediction Page", layout="centered")
st.title("2026 IndyCar Season")

Base_dir = Path(__file__).resolve().parent.parent / "IndyCar"

# Paths to models, datset and model prediction results
PRE_QUALY_MODEL = Base_dir/"models"/"indycar_lgbm_cat_prequaly_model_v1"
POST_QUALY_MODEL = Base_dir/"models"/"indycar_cat_lgbm_br_postqualy_model_v3"
DATASET = Base_dir/"datasets"/"IndyCar_dataset_v18.csv"
PREDICTIONS = Base_dir/"predictions"/"model_compare.xlsx"

# Load requested model
@st.cache_resource
def load_requested_model(model: str):
    return load_model(model)

# Load needed data
@st.cache_data
def load_feature_cols(data: str, is_postqualy: bool):
    df = pd.read_csv(data)
    drop_cols = ["DriverName", "TeamName", "CarEngine", "EventName", "Track", "EventTrackType",
                 "EventDate", "EventDateFormatted", "EventID", "Era", "Status", "StatusID",
                 "PositionFinish", "NormalizedPositionFinish"]

    if not is_postqualy:
        drop_cols.append("PositionStart")

    feature_row = df.drop(columns=drop_cols, errors="ignore").columns.tolist()
    return feature_row

@st.cache_data
def load_averages(data: str):
    df = pd.read_csv(data)
    return df.select_dtypes(include=[np.number]).mean().to_dict()

@st.cache_data
def load_field_size(data: str) -> int:
    df = pd.read_csv(data)
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    return int(df.sort_values(["EventDate", "EventID"])["FieldSize"].tail(1).values[0])

@st.cache_data
def load_stats(data: str) -> dict:
    df =pd.read_csv(data)
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    for col in ["DriverID", "TeamID", "EngineID", "TrackID", "EventTrackTypeID"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    df_sorted = df.sort_values(["EventDate", "EventID"])
    df_rookies = df_sorted[df_sorted["Rookie"] == 1]

    return {
        # Driver Data
        "driver" : df_sorted.groupby("DriverID", sort=False).tail(1).set_index("DriverID").to_dict("index"),
        "driver_track": df_sorted.groupby(["DriverID", "TrackID"], sort=False).tail(1).set_index(["DriverID", "TrackID"]).to_dict("index"),
        "driver_tracktype": df_sorted.groupby(["DriverID", "EventTrackTypeID"], sort=False).tail(1).set_index(["DriverID", "EventTrackTypeID"]).to_dict("index"),

        # Rookie Data
        "rookie_avgs": df_rookies.select_dtypes(include=[np.number]).mean().to_dict(),
        "rookie_team": df_rookies.groupby("TeamID", sort=False).tail(1).set_index("TeamID").to_dict("index"),
        "rookie_team_track": df_rookies.groupby(["TeamID", "TrackID"], sort=False).tail(1).set_index(["TeamID", "TrackID"]).to_dict("index"),
        "rookie_team_tracktype": df_rookies.groupby(["TeamID", "EventTrackTypeID"], sort=False).tail(1).set_index(["TeamID", "EventTrackTypeID"]).to_dict("index"),

        # Teams Data
        "team": df_sorted.groupby("TeamID", sort=False).tail(1).set_index("TeamID").to_dict("index"),
        "team_track": df_sorted.groupby(["TeamID", "TrackID"], sort=False).tail(1).set_index(["TeamID", "TrackID"]).to_dict("index"),
        "team_tracktype": df_sorted.groupby(["TeamID", "EventTrackTypeID"], sort=False).tail(1).set_index(["TeamID", "EventTrackTypeID"]).to_dict("index"),

        # Engine Data
        "engine": df_sorted.groupby("EngineID", sort=False).tail(1).set_index("EngineID").to_dict("index"),
        "engine_track": df_sorted.groupby(["EngineID", "TrackID"], sort=False).tail(1).set_index(["EngineID", "TrackID"]).to_dict("index"),
        "engine_tracktype": df_sorted.groupby(["EngineID", "EventTrackTypeID"], sort=False).tail(1).set_index(["EngineID", "EventTrackTypeID"]).to_dict("index")
    }

@st.cache_data
def load_predictions(filepath: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, sheet_name=sheet)
    df.columns = df.columns.str.strip()
    return df

# Drivers Map- name: {id, team_id, team_name, rookie(bool)}
DRIVERS_MAP = {
    "Marcus Armstrong": {"id": 4948, "team_id": 19, "team_name": "Meyer Shank Racing", "rookie": False},
    "Caio Collet": {"id": 4983, "team_id": 1, "team_name": "A.J. Foyt Enterprises", "rookie": True},
    "Scott Dixon": {"id": 3628, "team_id": 9, "team_name": "Chip Ganassi Racing", "rookie": False},
    "Marcus Ericsson": {"id": 4905, "team_id": 3,  "team_name": "Andretti Global", "rookie": False},
    "Santino Ferrucci": {"id": 4897, "team_id": 1, "team_name": "A.J. Foyt Enterprises", "rookie": False},
    "Louis Foster": {"id": 4949, "team_id": 24, "team_name": "Rahal Letterman Lanigan Racing", "rookie": False},
    "Romain Grosjean": {"id": 4935, "team_id": 11, "team_name": "Dale Coyne Racing", "rookie": False},
    "Dennis Hauger": {"id": 4982, "team_id": 11, "team_name": "Dale Coyne Racing", "rookie": True},
    "Kyle Kirkwood": {"id": 4623, "team_id": 3, "team_name": "Andretti Global", "rookie": False},
    "Christian Lundgaard": {"id": 4938, "team_id": 4, "team_name": "Arrow McLaren", "rookie": False},
    "David Malukas": {"id": 4636, "team_id": 28, "team_name": "Team Penske", "rookie": False},
    "Scott McLaughlin": {"id": 4932, "team_id": 28, "team_name": "Team Penske", "rookie": False},
    "Josef Newgarden": {"id": 4215, "team_id": 28, "team_name": "Team Penske", "rookie": False},
    "Pato O'Ward": {"id": 4559, "team_id": 4, "team_name": "Arrow McLaren", "rookie": False},
    "Alex Palou": {"id": 4931, "team_id": 9, "team_name": "Chip Ganassi Racing", "rookie": False},
    "Will Power": {"id": 3667, "team_id": 3,  "team_name": "Andretti Global",  "rookie": False},
    "Graham Rahal": {"id": 3668, "team_id": 24, "team_name": "Rahal Letterman Lanigan Racing",  "rookie": False},
    "Christian Rasmussen": {"id": 4920, "team_id": 15, "team_name": "Ed Carpenter Racing", "rookie": False},
    "Sting Ray Robb": {"id": 4613, "team_id": 16, "team_name": "Juncos Hollinger Racing", "rookie": False},
    "Felix Rosenqvist": {"id": 4588, "team_id": 19, "team_name": "Meyer Shank Racing", "rookie": False},
    "Alexander Rossi": {"id": 4587, "team_id": 15, "team_name": "Ed Carpenter Racing", "rookie": False},
    "Mick Schumacher": {"id": 4984, "team_id": 24, "team_name": "Rahal Letterman Lanigan Racing", "rookie": True},
    "Nolan Siegel": {"id": 4915, "team_id": 4, "team_name": "Arrow McLaren", "rookie": False},
    "Kyffin Simpson": {"id": 4944, "team_id": 9, "team_name": "Chip Ganassi Racing", "rookie": False},
    "Rinus VeeKay": {"id": 4614, "team_id": 16, "team_name": "Juncos Hollinger Racing", "rookie": False},
}

# Teams Map- team_id: {name, engine, engine_id}
TEAMS_MAP = {
    1: {"name": "A.J. Foyt Enterprises", "engine": "Chevrolet", "engine_id": 0},
    3: {"name": "Andretti Global", "engine": "Honda", "engine_id": 1},
    4: {"name": "Arrow McLaren", "engine": "Chevrolet", "engine_id": 0},
    9: {"name": "Chip Ganassi Racing", "engine": "Honda", "engine_id": 1},
    11: {"name": "Dale Coyne Racing", "engine": "Honda", "engine_id": 1},
    15: {"name": "Ed Carpenter Racing", "engine": "Chevrolet", "engine_id": 0},
    16: {"name": "Juncos Hollinger Racing", "engine": "Chevrolet", "engine_id": 0},
    19: {"name": "Meyer Shank Racing", "engine": "Honda", "engine_id": 1},
    24: {"name": "Rahal Letterman Lanigan Racing", "engine": "Honda", "engine_id": 1},
    28: {"name": "Team Penske", "engine": "Chevrolet", "engine_id": 0},
}

# Trakcs Map- name: {id, type, type_id, is_new(bool)}
TRACKS_MAP ={
"Streets of St. Petersburg": {"id": 21, "type": "Street", "type_id": 2, "is_new": False},
    "Phoenix Raceway": {"id": 10, "type": "Oval", "type_id": 0, "is_new": False},
    "Streets of Arlington": {"id": 29,"type": "Street", "type_id": 2, "is_new": False},
    "Barber Motorsports Park": {"id": 1, "type": "Road", "type_id": 1, "is_new": False},
    "Streets of Long Beach": {"id": 18, "type": "Street", "type_id": 2, "is_new": False},
    "Indianapolis Motor Speedway (Road)": {"id": 3, "type": "Road", "type_id": 1, "is_new": False},
    "Indianapolis Motor Speedway (Oval)": {"id": 4, "type": "Oval", "type_id": 0, "is_new": False},
    "Streets of Detroit": {"id": 17, "type": "Street", "type_id": 2, "is_new": False},
    "World Wide Technology Raceway": {"id": 28, "type": "Oval", "type_id": 0, "is_new": False},
    "Road America": {"id": 14, "type": "Road", "type_id": 1, "is_new": False},
    "Mid-Ohio Sports Car Course": {"id": 6, "type": "Road", "type_id": 1, "is_new": False},
    "Nashville Superspeedway": {"id": 8, "type": "Oval", "type_id": 0, "is_new": False},
    "Portland International Raceway": {"id": 12, "type": "Road", "type_id": 1, "is_new": False},
    "Streets of Markham": {"id": None,"type": "Street", "type_id": 2, "is_new": True},
    "Streets of Washington": {"id": None,"type": "Street", "type_id": 2, "is_new": True},
    "WeatherTech Raceway Laguna Seca": {"id": 27, "type": "Road", "type_id": 1, "is_new": False},
    "Milwaukee Mile": {"id": 7, "type": "Oval", "type_id": 0, "is_new": False},
}

# Helper function to create feature row for each driver
def populate_feature_row(driver_name, track_info, is_postqualy, position_start, FEATURES, DATASET_AVGS, STATS):
    driver_info = DRIVERS_MAP[driver_name]
    team_id = driver_info["team_id"]
    track_id = track_info['id']
    type_id = track_info["type_id"]
    is_new = track_info["is_new"]
    engine_id = TEAMS_MAP[team_id]["engine_id"]

    # Features baseline (Averages)
    features = {col: DATASET_AVGS.get(col, 0) for col in FEATURES}

    # Rookie indicator
    features["Rookie"] = int(driver_info["rookie"])

    # Populates Driver specific features
    if driver_info["id"] and not driver_info["rookie"]:
        driver_row = STATS["driver"].get(driver_info["id"], None)
        track_row = STATS["driver_track"].get((driver_info["id"], track_id), None) if not is_new else None
        tracktype_row = STATS["driver_tracktype"].get((driver_info["id"], type_id), None)

        if driver_row:
            for col in ["DRFAvg", "DNFRate", "DriverElo", "DriverRitmo"]:
                if col in FEATURES and col in driver_row:
                    features[col] = driver_row[col]

        row = track_row or tracktype_row or driver_row
        if row:
            for col in ["DTAvg", "TDNFRate","DriverTElo"]:
                if col in FEATURES and col in row:
                    features[col] = row[col]

        row = tracktype_row or driver_row
        if row:
            for col in ["DTTAvg", "DriverTTElo"]:
                if col in FEATURES and col in row:
                    features[col] = row[col]

        if "DriverID" in FEATURES:
            features["DriverID"] = int(driver_info["id"])

    else:
        # Rookies
        driver_row = STATS["rookie_avgs"]
        track_row = STATS["rookie_team_track"].get((team_id, track_id), None) if not is_new else None
        tracktype_row = STATS["rookie_team_tracktype"].get((team_id, type_id), None)
        team_row = STATS["rookie_team"].get(team_id, None)

        if team_row:
            for col in ["DRFAvg", "DNFRate", "DriverElo", "DriverRitmo"]:
                if col in FEATURES and col in team_row:
                    features[col] = team_row[col]

        row = track_row or tracktype_row or driver_row
        if row:
            for col in ["DTAvg", "TDNFRate","DriverTElo"]:
                if col in FEATURES and col in row:
                    features[col] = row[col]

        row = tracktype_row or driver_row
        if row:
            for col in ["DTTAvg", "DriverTTElo"]:
                if col in FEATURES and col in row:
                    features[col] = row[col]

        if "DriverID" in FEATURES:
            features["DriverID"] = int(STATS["rookie_avgs"].get("DriverID", 0))

    # Populates Team specific features
    team_row = STATS["team"].get(team_id, None)
    track_row = STATS["team_track"].get((team_id, track_id), None) if not is_new else None
        
    if team_row:
        for col in ["TRP", "TeamDNFRate", "TeamElo", "TeamRitmo"]:
            if col in FEATURES and col in team_row:
                features[col] = team_row[col]
    row = track_row or team_row
    if row:
        for col in ["TTP", "TeamTElo"]:
            if col in FEATURES and col in row:
                features[col] = row[col]

    if "TeamID" in FEATURES:
        features["TeamID"] = int(team_id)

    # Populates Engine specific features
    engine_row = STATS["engine"].get(engine_id, None)
    track_row = STATS["engine_track"].get((engine_id, track_id), None) if not is_new else None
    tracktype_row = STATS["engine_tracktype"].get((engine_id, type_id), None)

    if engine_row:
        for col in ["EngineElo"]:
            if col in FEATURES and col in engine_row:
                features[col] = engine_row[col]

    row = track_row or tracktype_row or engine_row
    if row:
        for col in ["EngineTElo"]:
            if col in FEATURES and col in row:
                features[col] = row[col]

    row = tracktype_row or engine_row
    if row:
        for col in ["EngineTTElo"]:
            if col in FEATURES and col in row:
                features[col] = row[col]

    if "EngineID" in FEATURES:
        features["EngineID"] = int(engine_id)

    # Populates Track features
    if "TrackID" in FEATURES:
        features["TrackID"] = int(track_id) if track_id is not None else 0

    if "EventTrackTypeID" in FEATURES:
        features["EventTrackTypeID"] = int(type_id)

    if is_postqualy and position_start is not None:
        if "PositionStart" in FEATURES:
            features["PositionStart"] = int(position_start)

    if "EraID" in FEATURES:
        features["EraID"] = 2

    return features

# Helper function to Denormalize postions for visual purposes
def denormalizer(normalized: float, field_size: int) -> int:
    raw = normalized * (field_size - 1) +1
    return  int(round(np.clip(raw, 1, field_size)))

# Helper functions for the results tab
def error_color(val):
    if val in ("", "-"):
        return ""

    try:
        n = int(val)
        if abs(n) <= 2:
            return "color: green"
        elif abs(n) <= 5:
            return "color: orange"
        return "color: red"
    except:
        return ""

def fmt_score(val):
    return f"{val:.2f}" if pd.notna(val) else "-"

def fmt_pos(val):
    return f"P{int(val)}" if pd.notna(val) else "-"


# UI 
st.page_link("sportssystem_homepage.py", label="Back to Homepage")
tab_sim, tab_results = st.tabs(["Race Simulator", "Results"])

# Tab 1 - Race Simulator

with tab_sim:
    # Model Selection
    st.subheader("Prediction Phase")
    mode = st.radio("Select prediction phase:", ["Pre-Qualy", "Post-Qualy"], horizontal = True)
    is_postqualy = mode.startswith("Post")

    # Load selected model and stats
    model_path = POST_QUALY_MODEL if is_postqualy else PRE_QUALY_MODEL
    model = load_requested_model(str(model_path))
    FEATURES = load_feature_cols(str(DATASET), is_postqualy)
    DATASET_AVGS = load_averages(str(DATASET))
    FIELD_SIZE = load_field_size(str(DATASET))
    STATS = load_stats(str(DATASET))

    st.subheader("Race Simulator")
    st.markdown("Predicts all drivers finishing position and ranks them")

    track_name_sim = st.selectbox("Track", list(TRACKS_MAP.keys()), key = "sim_track")
    track_info_sim = TRACKS_MAP[track_name_sim]

    if track_info_sim["is_new"]:
        st.warning(f"**{track_name_sim}** is a new track. Therefore, predictions will be based on Track Type metrics only.")

    # Post-Quali: grid position for all drivers
    grid_positions = {}
    if is_postqualy:
        st.markdown("**Enter Qualifying grid positions:**")
        cols = st.columns(3)
        for i, driver, in enumerate(DRIVERS_MAP.keys()):
            with cols[i % 3]:
                grid_positions[driver] = st.number_input(
                    driver, min_value = 1, max_value = 33,
                    value = i + 1, key = f"grid_{driver}"
                )

    if st.button("Predict", use_container_width = True, key = "sim_predict"):
        try:
            all_rows = []
            driver_names = []

            for driver in DRIVERS_MAP.keys():
                position = grid_positions.get(driver) if is_postqualy else None
                features = populate_feature_row(driver, track_info_sim, is_postqualy, position, FEATURES, DATASET_AVGS, STATS)
                all_rows.append(features)
                driver_names.append(driver)

            input_df = pd.DataFrame(all_rows, columns = FEATURES)

            results = predict_model(model, data=input_df)
            results["Driver"] = driver_names
            results["Raw Score"] = results["prediction_label"].apply(lambda x: round(x * (FIELD_SIZE - 1) + 1, 2))
            results["Position"] = results["prediction_label"].rank(method="first").astype(int)
            results = results.sort_values("Position").reset_index(drop=True)

            display_df = results[["Position", "Driver", "Raw Score"]].set_index("Position")
            st.dataframe(display_df, use_container_width=True)

        except Exception as e:
            st.error(f"Simulation error: {e}")
            st.exception(e)

# Tab 2 - Results Comparison
with tab_results:
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
        selected_track = st.selectbox("Race:", tracks)

    df_race = df[df["Track"] == selected_track].copy().sort_values("ModelRank").reset_index(drop=True)

    if df_race.empty:
        st.warning("No predictions found for this track.")
        st.stop()


    # Track metrics
    has_results = df_race["ActualFinish"].notna().any()
    track_type = df_race["TrackType"].iloc[0] if "TrackType" in df_race.columns else "-"
    model_mae = df_race["ModelMAE"].iloc[0] if "ModelMAE" in df_race.columns and pd.notna(df_race["ModelMAE"].iloc[0]) else None

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
        model_df = df_race[["ModelRank", "Driver", "RawScore", "Low", "High", "ActualFinish"]].copy()
        model_df = model_df.sort_values("ModelRank").reset_index(drop=True)
        model_df.columns = ["Position", "Driver", "Score", "Low", "High", "ActualFinish"]
        model_df["Position"] = model_df["Position"].astype(int)

        if has_results:
            model_df["Actual"] = model_df["ActualFinish"].apply(fmt_pos)
            model_df["Error"] = model_df.apply(
                lambda r: (f"+{int(r['ActualFinish'] - r['Position'])}"
                        if r['ActualFinish'] - r['Position'] > 0
                        else str(int(r['ActualFinish'] - r['Position'])))
                if pd.notna(r['ActualFinish']) else "-", axis=1
            )
        
        model_df = model_df.drop(columns=["ActualFinish"])

        formatted = model_df.style.format({"Score": "{:.2f}", "Low": "{:.2f}", "High": "{:.2f}"})
        if has_results:
            formatted = formatted.applymap(error_color, subset=["Error"])

        st.dataframe(formatted, hide_index=True, use_container_width=True)

    with my_col:
        st.caption("My Predictions")
        my_df = df_race[["MyRank", "Driver", "MyRawScore"]].copy()
        my_df = my_df.sort_values("MyRank").reset_index(drop=True)
        my_df.columns = ["Position", "Driver", "Score"]
        my_df["Position"] = my_df["Position"].astype(int)

        st.dataframe(my_df.style.format({"Score": "{:.2f}"}), hide_index=True, use_container_width=True)

    st.divider()

    # Model vs Mine Graph
    st.subheader("Model Position Prediction vs My Position Prediction")

    chart_df = df_race[["Driver", "ModelRank", "MyRank"]].dropna().copy()
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
    diag = alt.Chart(diag_df).mark_line(color="gray", strokeDash=[4,4], opacity=.45).encode(x="x:Q", y="y:Q")

    st.altair_chart(graph + diag, use_container_width=True)


    # Actual vs Predicted (After race is complete)
    if has_results:
        st.divider()
        st.subheader("Actual vs Predicted finish")
        result_df = df_race[["Driver", "ModelRank", "ActualFinish"]].dropna().copy()
        result_df["ModelRank"] = result_df["ModelRank"].astype(int)
        result_df["ActualFinish"] = result_df["ActualFinish"].astype(int)
        result_df["Error"] = (result_df["ActualFinish"] - result_df["ModelRank"]).abs()
        result_df["ErrorColor"] = result_df["Error"].apply(lambda e: "Good" if e <= 2 else ("Close" if e <= 5 else "Off"))
        result_df = result_df.sort_values("ActualFinish").reset_index(drop=True)

        bar = alt.Chart(result_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Driver:N", sort=alt.EncodingSortField(field="ActualFinish", order="ascending"), title=""),
            y=alt.Y("Error:Q", title="Position Error"),
            color=alt.Color("ErrorColor:N", scale=alt.Scale(
                domain= ["Good", "Close", "Off"],
                range = ["#3B6D11", "#F5DA27", "#A32D2D"]
                ),legend=alt.Legend(title="Accuracy")),
            tooltip=["Driver:N",
                    alt.Tooltip("ModelRank:Q", title="Predicted"),
                    alt.Tooltip("ActualFinish:Q", title="Actual"),
                    alt.Tooltip("Error:Q", title="Error")]).properties(height=380)

        st.altair_chart(bar, use_container_width=True)

        computed_mae = result_df["Error"].mean()
        best = result_df.loc[result_df["Error"].idxmin(), "Driver"]
        worst = result_df.loc[result_df["Error"].idxmax(), "Driver"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Actual MAE", f"{computed_mae:.2f}")
        c2.metric("Best prediction", best)
        c3.metric("Worst prediction", worst)