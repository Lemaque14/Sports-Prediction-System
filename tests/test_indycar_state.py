"""Replay check: the up-to-date state built from every race before race E must
equal the pre-race features the dataset stores for race E."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.indycar import BASE_ELO, build_current_state, prepare

DATASET = Path(__file__).resolve().parent.parent / "IndyCar" / "datasets" / "IndyCar_dataset_v25.csv"
N_EVENTS = 12  # replay the most recent races


@pytest.fixture(scope="module")
def df():
    return prepare(pd.read_csv(DATASET))


def _event_starts(df):
    starts = df.drop_duplicates("EventID")
    return starts.index[-N_EVENTS:]


def _check(expected, actual, label):
    if pd.isna(expected):
        assert actual is None or pd.isna(actual), label
    else:
        assert actual == pytest.approx(expected, abs=1e-5), label  # CSV keeps 6 decimals


@pytest.mark.parametrize("pos", range(N_EVENTS))
def test_state_before_race_matches_dataset(df, pos):
    start = _event_starts(df)[pos]
    history = df.loc[: start - 1]
    event = df[df["EventID"] == df.loc[start, "EventID"]]
    state = build_current_state(history)
    track, ttype = int(event["TrackID"].iloc[0]), int(event["EventTrackTypeID"].iloc[0])

    for _, r in event.iterrows():
        d, t, e = int(r.DriverID), int(r.TeamID), int(r.EngineID)
        drv = state["driver"].get(d, {})
        drv_tr = state["driver_track"].get((d, track), {})
        drv_tt = state["driver_tracktype"].get((d, ttype), {})
        for col in ["DRFAvg", "DNFRate", "DriverRitmo"]:
            _check(r[col], drv.get(col), f"{col} driver {d}")
        _check(r.DriverElo, drv.get("DriverElo", BASE_ELO), f"DriverElo {d}")
        _check(r.DTAvg, drv_tr.get("DTAvg"), f"DTAvg {d}")
        _check(r.TDNFRate, drv_tr.get("TDNFRate"), f"TDNFRate {d}")
        _check(r.DriverTElo, drv_tr.get("DriverTElo", BASE_ELO), f"DriverTElo {d}")
        _check(r.DTTAvg, drv_tt.get("DTTAvg"), f"DTTAvg {d}")
        _check(r.DriverTTElo, drv_tt.get("DriverTTElo", BASE_ELO), f"DriverTTElo {d}")

        tm = state["team"].get(t, {})
        for col in ["TRP", "TeamDNFRate", "TeamRitmo"]:
            _check(r[col], tm.get(col), f"{col} team {t}")
        _check(r.TeamElo, tm.get("TeamElo", BASE_ELO), f"TeamElo {t}")
        _check(r.TeamTElo, state["team_track"].get((t, track), {}).get("TeamTElo", BASE_ELO), f"TeamTElo {t}")
        _check(r.TTP, state["team_tracktype"].get((t, ttype), {}).get("TTP"), f"TTP {t}")

        _check(r.EngineElo, state["engine"].get(e, {}).get("EngineElo", BASE_ELO), f"EngineElo {e}")
        _check(r.EngineTElo, state["engine_track"].get((e, track), {}).get("EngineTElo", BASE_ELO), f"EngineTElo {e}")
        _check(r.EngineTTElo, state["engine_tracktype"].get((e, ttype), {}).get("EngineTTElo", BASE_ELO), f"EngineTTElo {e}")

    first = event.iloc[0]
    trk = state["track"].get(track, {})
    for col in ["TrackAvgCautions", "TrackAvgCautionLaps", "TrackAvgSpeed", "TrackAvgPitStops"]:
        _check(first[col], trk.get(col), col)
    tt = state["tracktype"].get(ttype, {})
    for col in ["TrackTypeAvgCautions", "TrackTypeAvgCautionLaps"]:
        _check(first[col], tt.get(col), col)

    # Team pit-stop averages are taken over car rows, so only each team's first car
    # in the race sees exactly the history before the race.
    for _, r in event.drop_duplicates("TeamID").iterrows():
        row = state["team_track"].get((int(r.TeamID), track), {})
        for col in ["TeamTrackAvgPitStops", "TeamTrackAvgPittedCautionPCT"]:
            _check(r[col], row.get(col), f"{col} team {r.TeamID}")


def test_state_includes_latest_race(df):
    """The app's state must reflect the most recent result (the old app did not)."""
    state = build_current_state(df)
    for driver_id, rows in df.groupby("DriverID"):
        expected = rows["NormalizedPositionFinish"].tail(5).mean()
        assert state["driver"][int(driver_id)]["DRFAvg"] == pytest.approx(expected)
    assert np.isfinite([row["DriverElo"] for row in state["driver"].values()]).all()
