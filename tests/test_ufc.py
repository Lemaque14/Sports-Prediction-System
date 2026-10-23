from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features import ufc

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "UFC" / "datasets"
MODELS = ROOT / "UFC" / "models"


@pytest.fixture(scope="module")
def fights():
    return pd.read_csv(DATA / "UFC_fights_v1.csv", parse_dates=["Date"])


@pytest.fixture(scope="module")
def fighters():
    return pd.read_csv(DATA / "UFC_fighters_v1.csv", parse_dates=["DOB"])


@pytest.fixture(scope="module")
def feats(fights, fighters):
    return ufc.history_features(fights, fighters)


def test_parsing_helpers():
    assert ufc._fight_seconds(2, 152, "3 Rnd (5-5-5)") == 452
    assert ufc._fight_seconds(1, 30, "No Time Limit") == 30
    assert ufc._weight_class("UFC Light Heavyweight Title Bout") == ("Light Heavyweight", 205)
    assert ufc._method("TKO - Doctor's Stoppage") == "KO"
    assert ufc._method("Decision - Split") == "DEC"
    assert ufc._method("Overturned") == "OTHER"
    assert ufc._inches(pd.Series(["5' 11\"", "72.0\"", "--"])).tolist()[:2] == [71, 72]


def test_one_row_per_fighter_per_fight(fights):
    assert (fights.groupby("FightID").size() == 2).all()
    assert not fights["WeightClass"].str.contains("Women").any()


@pytest.mark.parametrize("back", [1, 2, 3])
def test_live_features_match_training_rows(fights, fighters, feats, back):
    """The app's features for a fighter's next fight equal the training row that fight gets."""
    day = pd.Timestamp(sorted(fights["Date"].unique())[-back])
    state = ufc.current_state(fights, fighters, day)
    rows = feats[feats["Date"] == day].set_index("Fighter")
    common = rows.index.intersection(state.index)
    assert len(common) > 0
    np.testing.assert_allclose(rows.loc[common, ufc.FEATURES].astype(float),
                               state.loc[common, ufc.FEATURES].astype(float), equal_nan=True)


def test_live_features_for_a_future_date(fights, fighters):
    """The app's usual case: a fight date after the last one in the data, entered as a plain date."""
    last_day = fights["Date"].max()
    later = ufc.current_state(fights, fighters, (last_day + pd.Timedelta(days=30)).date())
    sooner = ufc.current_state(fights, fighters, last_day + pd.Timedelta(days=1))
    assert len(later) == fights["Fighter"].nunique()
    same = [c for c in ufc.FEATURES if c not in ("Age", "DaysSinceLast")]
    np.testing.assert_allclose(later[same].astype(float), sooner.loc[later.index, same].astype(float), equal_nan=True)
    assert (later["DaysSinceLast"] - sooner.loc[later.index, "DaysSinceLast"] == 29).all()


def test_features_ignore_the_fight_itself(fights, fighters, feats):
    """Changing a fight's result and stats can't change that fight's own features."""
    fight_id = fights.loc[fights["Date"] >= "2025-06-01", "FightID"].iloc[0]
    changed = fights.copy()
    mask = changed["FightID"] == fight_id
    changed.loc[mask, "Result"] = changed.loc[mask, "Result"].map({"W": "L", "L": "W"}).fillna("NC")
    changed.loc[mask, ["SigLanded", "TDLanded", "KD"]] = 99
    after = ufc.history_features(changed, fighters)
    cols = ufc.FEATURES
    before_rows = feats[feats["FightID"] == fight_id].set_index("Fighter")[cols]
    after_rows = after[after["FightID"] == fight_id].set_index("Fighter")[cols]
    pd.testing.assert_frame_equal(before_rows, after_rows)
    fighter = before_rows.index[0]
    later = feats[(feats["Fighter"] == fighter) & (feats["Date"] > feats.loc[feats["FightID"] == fight_id, "Date"].iloc[0])]
    if len(later):  # the fighter's next fight does see the change
        assert not np.allclose(later.iloc[0][["Elo"]].astype(float),
                               after.loc[later.index[0], ["Elo"]].astype(float))


def test_training_rows_are_balanced_and_mirrored(fights, feats):
    table = ufc.fight_table(fights, feats)
    rows = ufc.winner_training_rows(table)
    assert rows["AWon"].mean() == 0.5
    swapped = ufc.swap(table.head(50))
    np.testing.assert_allclose(swapped["D_Elo"], -table.head(50)["D_Elo"].to_numpy())
    rounds = ufc.round_training_rows(table)
    assert set(rounds["Outcome"]) <= set(ufc.OUTCOMES)


def test_outcome_table_adds_up_and_respects_rounds():
    how = np.array([0.3, 0.2, 0.1, 0.05, 0.05, 0.3])
    three = ufc.outcome_table(0.6, how, how[::-1], rounds=3)
    assert list(three.columns) == ["R1", "R2", "R3", "DEC"]
    assert three.to_numpy().sum() == pytest.approx(1)
    assert three.loc["A"].sum() == pytest.approx(0.6)
    five = ufc.outcome_table(0.6, how, how[::-1], rounds=5)
    assert list(five.columns) == ufc.OUTCOMES
    swapped = ufc.outcome_table(0.4, how[::-1], how, rounds=3)
    np.testing.assert_allclose(three.loc["A"], swapped.loc["B"])


def test_symmetric_win_probability():
    assert ufc.symmetric_win_probability(0.7, 0.4) == pytest.approx(0.65)
    assert ufc.symmetric_win_probability(0.4, 0.7) == pytest.approx(0.35)


@pytest.mark.skipif(not (MODELS / "ufc_winner_model_v1.pkl").exists(), reason="models not trained")
def test_saved_models_give_a_consistent_table(fights, fighters):
    pycaret = pytest.importorskip("pycaret.classification")
    winner, rounds_model = (pycaret.load_model(str(MODELS / name)) for name in ["ufc_winner_model_v1", "ufc_round_model_v1"])
    state = ufc.current_state(fights, fighters, fights["Date"].max() + pd.Timedelta(days=1))
    pair = state.sort_values("LastFight").tail(2)
    a, b = pair.iloc[[0]].reset_index(), pair.iloc[[1]].reset_index()
    context = pd.DataFrame([{"ScheduledRounds": 3, "Title": 0, "WeightLbs": 155}])
    rows_ab, rows_ba = ufc.matchup_rows(a, b, context), ufc.matchup_rows(b, a, context)

    def win(rows):
        return pycaret.predict_model(winner, data=rows, raw_score=True, verbose=False)["prediction_score_1"].to_numpy()

    def how(rows):
        cols = [f"prediction_score_{o}" for o in ufc.OUTCOMES]
        return pycaret.predict_model(rounds_model, data=rows, raw_score=True, verbose=False)[cols].to_numpy()[0]

    table = ufc.outcome_table(ufc.symmetric_win_probability(win(rows_ab), win(rows_ba))[0], how(rows_ab), how(rows_ba), 3)
    reverse = ufc.outcome_table(ufc.symmetric_win_probability(win(rows_ba), win(rows_ab))[0], how(rows_ba), how(rows_ab), 3)
    assert table.to_numpy().sum() == pytest.approx(1)
    np.testing.assert_allclose(table.loc["A"], reverse.loc["B"])
