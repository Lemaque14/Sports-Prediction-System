import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
from pathlib import Path
from pycaret.classification import load_model, predict_model

from features import ufc

st.set_page_config(page_title="UFC Prediction Page", layout="centered")

st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    [data-testid="stSidebarCollapsedControl"] {
        display: block;
    }
    </style>
""", unsafe_allow_html=True)

st.title("UFC Men's Divisions")

Base_dir = Path(__file__).resolve().parent.parent / "UFC"

# Paths to models, datasets and backtest results
WINNER_MODEL = Base_dir/"models"/"ufc_winner_model_v1"
ROUND_MODEL = Base_dir/"models"/"ufc_round_model_v1"
FIGHTS = Base_dir/"datasets"/"UFC_fights_v1.csv"
FIGHTERS = Base_dir/"datasets"/"UFC_fighters_v1.csv"
BACKTEST = Base_dir/"predictions"/"UFC_backtest_v1.csv"

ACTIVE_YEARS = 3   # fighters listed in the simulator: fought within this many years
OUTCOME_NAMES = {"R1": "Round 1", "R2": "Round 2", "R3": "Round 3", "R4": "Round 4", "R5": "Round 5", "DEC": "Decision"}
WEIGHT_CLASSES = {name: lbs for name, lbs in sorted(ufc.WEIGHT_LIMITS.items(), key=lambda x: x[1])}
WEIGHT_CLASSES["Catch/Open Weight"] = np.nan


@st.cache_resource
def load_requested_model(model: str):
    return load_model(model)

@st.cache_data
def load_history(fights_path: str, fighters_path: str):
    fights = pd.read_csv(fights_path, parse_dates=["Date"])
    fighters = pd.read_csv(fighters_path, parse_dates=["DOB"])
    return fights, fighters

@st.cache_data
def load_state(fights_path: str, fighters_path: str, as_of: date) -> pd.DataFrame:
    # Each fighter's features going into their next fight, built by the same code as the training data
    fights, fighters = load_history(fights_path, fighters_path)
    state = ufc.current_state(fights, fighters, pd.Timestamp(as_of))
    record = fights.groupby("Fighter")["Result"].value_counts().unstack(fill_value=0)
    for col in ["W", "L", "D", "NC"]:
        state[col] = record.reindex(state.index)[col] if col in record else 0
    return state

@st.cache_data
def load_backtest(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["Date"])


def predict_outcomes(state, fighter_a, fighter_b, rounds, title, weight_lbs):
    """Probability of each fighter winning in each round or by decision."""
    context = pd.DataFrame([{"ScheduledRounds": rounds, "Title": int(title), "WeightLbs": weight_lbs}])
    a, b = state.loc[[fighter_a]].reset_index(), state.loc[[fighter_b]].reset_index()
    rows_ab, rows_ba = ufc.matchup_rows(a, b, context), ufc.matchup_rows(b, a, context)

    winner_model = load_requested_model(str(WINNER_MODEL))
    round_model = load_requested_model(str(ROUND_MODEL))
    p_ab = predict_model(winner_model, data=rows_ab, raw_score=True, verbose=False)["prediction_score_1"]
    p_ba = predict_model(winner_model, data=rows_ba, raw_score=True, verbose=False)["prediction_score_1"]
    p_a = float(ufc.symmetric_win_probability(p_ab, p_ba)[0])

    score_cols = [f"prediction_score_{o}" for o in ufc.OUTCOMES]
    how_a = predict_model(round_model, data=rows_ab, raw_score=True, verbose=False)[score_cols].to_numpy()[0]
    how_b = predict_model(round_model, data=rows_ba, raw_score=True, verbose=False)[score_cols].to_numpy()[0]
    table = ufc.outcome_table(p_a, how_a, how_b, rounds)
    table.index = [fighter_a, fighter_b]
    return table


def fmt_pct(val):
    return f"{val:.1%}" if pd.notna(val) else "-"


# UI
st.page_link("sportssystem_homepage.py", label="Back to Homepage")
tab_sim, tab_results = st.tabs(["Fight Simulator", "Results"])

# Tab 1 - Fight Simulator
with tab_sim:
    st.subheader("Fight Simulator")
    st.markdown("Each fighter's chance of winning in every round or by decision")

    fight_date = st.date_input("Fight date", value=date.today(), key="ufc_date")
    state = load_state(str(FIGHTS), str(FIGHTERS), fight_date)
    active = state[state["LastFight"] >= pd.Timestamp(fight_date) - pd.DateOffset(years=ACTIVE_YEARS)]
    names = sorted(active.index)

    col1, col2 = st.columns(2)
    with col1:
        fighter_a = st.selectbox("Fighter", names, index=None, placeholder="Type to search..", key="ufc_a")
    with col2:
        fighter_b = st.selectbox("Opponent", [n for n in names if n != fighter_a], index=None,
                                 placeholder="Type to search..", key="ufc_b")

    col3, col4, col5 = st.columns(3)
    with col3:
        default_class = state.loc[fighter_a, "LastWeightClass"] if fighter_a else "Lightweight"
        classes = list(WEIGHT_CLASSES)
        weight_class = st.selectbox("Weight class", classes,
                                    index=classes.index(default_class) if default_class in classes else 0,
                                    key="ufc_class")
    with col4:
        rounds = st.radio("Rounds", [3, 5], horizontal=True, key="ufc_rounds")
    with col5:
        title = st.checkbox("Title fight", key="ufc_title")

    if st.button("Predict", use_container_width=True, key="ufc_predict"):
        if not fighter_a or not fighter_b:
            st.warning("Pick both fighters")
        else:
            try:
                table = predict_outcomes(state, fighter_a, fighter_b, rounds, title, WEIGHT_CLASSES[weight_class])
                totals = table.sum(axis=1)

                st.divider()
                c1, c2 = st.columns(2)
                c1.metric(fighter_a, fmt_pct(totals[fighter_a]))
                c1.progress(float(totals[fighter_a]))
                c2.metric(fighter_b, fmt_pct(totals[fighter_b]))
                c2.progress(float(totals[fighter_b]))

                display = table.rename(columns=OUTCOME_NAMES)
                display["Total"] = totals
                st.dataframe(display.style.format(fmt_pct), use_container_width=True)

                best_fighter, best_outcome = table.stack().idxmax()
                st.markdown(f"Most likely single outcome: **{best_fighter}** "
                            f"{'by decision' if best_outcome == 'DEC' else 'in ' + OUTCOME_NAMES[best_outcome]} "
                            f"({table.loc[best_fighter, best_outcome]:.1%})")

                chart_df = table.rename(columns=OUTCOME_NAMES).reset_index(names="Fighter").melt(
                    id_vars="Fighter", var_name="Outcome", value_name="Probability")
                chart = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X("Outcome:N", sort=list(display.columns[:-1]), title=""),
                    xOffset="Fighter:N",
                    y=alt.Y("Probability:Q", axis=alt.Axis(format=".0%")),
                    color=alt.Color("Fighter:N", legend=alt.Legend(orient="bottom")),
                    tooltip=["Fighter:N", "Outcome:N", alt.Tooltip("Probability:Q", format=".1%")],
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

                st.subheader("Tale of the tape")
                compare = state.loc[[fighter_a, fighter_b]]

                def num(val, digits=0):
                    return f"{val:.{digits}f}" if pd.notna(val) else "-"

                # Text throughout: each column mixes kinds of values once transposed
                tape = pd.DataFrame({
                    "UFC record (W-L-D)": [f"{int(r.W)}-{int(r.L)}-{int(r.D)}" for r in compare.itertuples()],
                    "Elo": [num(v) for v in compare["Elo"]],
                    "Age": [num(v, 1) for v in compare["Age"]],
                    "Height (in)": [num(v) for v in compare["HeightIn"]],
                    "Reach (in)": [num(v) for v in compare["ReachIn"]],
                    "Stance": compare["Stance"].fillna("-").tolist(),
                    "Sig. strikes landed / min": [num(v, 2) for v in compare["SigLandedPerMin"]],
                    "Sig. strikes absorbed / min": [num(v, 2) for v in compare["SigAbsorbedPerMin"]],
                    "Takedowns / 15 min": [num(v, 2) for v in compare["TDPer15"]],
                    "Days since last fight": [num(v) for v in compare["DaysSinceLast"]],
                }, index=compare.index).T
                st.dataframe(tape, use_container_width=True)

                download = display.reset_index(names="Fighter")
                download.insert(0, "FightDate", fight_date)
                st.download_button("Download this prediction (CSV)", download.to_csv(index=False),
                                   file_name=f"ufc_{fighter_a}_vs_{fighter_b}.csv".replace(" ", "_"))

            except Exception as e:
                st.error(f"Prediction error: {e}")

# Tab 2 - Results: every 2025-26 fight, predicted by models trained on fights up to 2024 only
with tab_results:
    try:
        bt = load_backtest(str(BACKTEST))
    except Exception as e:
        st.error(f"Could not load results file: {e}")
        st.stop()

    st.caption("Models trained on fights through 2024 only, then asked about every men's fight since. "
               "Baseline: Elo rating for who wins, historical round shares for when.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Winner picked", fmt_pct(bt["PickCorrect"].mean()), f"Elo {bt['EloPickCorrect'].mean():.1%}",
              delta_color="off")
    c2.metric("Log loss, full table", f"{bt['LogLoss'].mean():.3f}", f"baseline {bt['BaselineLogLoss'].mean():.3f}",
              delta_color="off")
    c3.metric("Fights", f"{len(bt)}")

    events = bt.sort_values("Date", ascending=False)["Event"].unique().tolist()
    selected = st.selectbox("Event", events, key="ufc_event")
    ev = bt[bt["Event"] == selected].copy()

    ev["Model pick"] = [f"{a if p >= 0.5 else b} ({max(p, 1 - p):.0%})"
                        for a, b, p in zip(ev["FighterA"], ev["FighterB"], ev["PWinA"])]
    ev["Most likely outcome"] = ev["TopCell"]
    ev["Result"] = ev["Actual"]
    ev["Correct"] = ev["PickCorrect"].map({True: "Yes", False: "No"})
    ev["Chance given to result"] = ev["PActual"].map(fmt_pct)
    view = ev[["FighterA", "FighterB", "Model pick", "Most likely outcome", "Result", "Correct", "Chance given to result"]]
    view = view.rename(columns={"FighterA": "Fighter", "FighterB": "Opponent"})
    st.dataframe(view.style.map(lambda v: "color: green" if v == "Yes" else ("color: red" if v == "No" else ""),
                                subset=["Correct"]),
                 hide_index=True, use_container_width=True)

    st.subheader("Accuracy by month")
    monthly = bt.assign(Month=bt["Date"].dt.to_period("M").dt.to_timestamp()).groupby("Month").agg(
        Model=("PickCorrect", "mean"), Elo=("EloPickCorrect", "mean")).reset_index().melt(
        id_vars="Month", var_name="Picker", value_name="Accuracy")
    line = alt.Chart(monthly).mark_line(point=True).encode(
        x=alt.X("Month:T", title=""), y=alt.Y("Accuracy:Q", axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0.3, 0.9])),
        color="Picker:N", tooltip=["Month:T", "Picker:N", alt.Tooltip("Accuracy:Q", format=".1%")],
    ).properties(height=260)
    st.altair_chart(line, use_container_width=True)
