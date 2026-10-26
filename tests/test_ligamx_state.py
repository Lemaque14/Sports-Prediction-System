from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features import ligamx

DATASET = Path(__file__).resolve().parent.parent / "LigaMX" / "datasets" / "LigaMX_dataset_v6.csv"


@pytest.fixture(scope="module")
def df():
    return ligamx.prepare(pd.read_csv(DATASET))


def _pairs(df):
    """Both rows of every match that appears twice, side by side."""
    two = df.groupby("match_id").filter(lambda m: len(m) == 2)
    first = two.groupby("match_id").nth(0).set_index("match_id")
    second = two.groupby("match_id").nth(1).set_index("match_id")
    return first, second.loc[first.index]


def test_both_rows_of_a_match_get_pre_match_elo(df):
    a, b = _pairs(df)
    assert np.allclose(a["TeamElo"], b["OponentElo"])
    assert np.allclose(a["OponentElo"], b["TeamElo"])


def test_both_rows_of_a_match_get_pre_match_h2h(df):
    a, b = _pairs(df)
    played = a["H2HWinRate"].notna()
    assert (played == b["H2HWinRate"].notna()).all()
    # (wins + losses + draws) / matches = 1 seen from either side, so win rates can't exceed 1 together
    assert ((a.loc[played, "H2HWinRate"] + b.loc[played, "H2HWinRate"]) <= 1 + 1e-9).all()


def test_second_row_no_longer_predicts_the_result(df):
    """With leakage, Elo picked the winner far more often on second rows (72.8% vs 60.9%)."""
    df = df.copy()
    df["pos"] = df.groupby("match_id").cumcount()
    decided = df[df["Result"].isin(["W", "L"])]
    hit = ((decided["EloDiff"] > 0) == (decided["Result"] == "W")).groupby(decided["pos"]).mean()
    assert abs(hit[0] - hit[1]) < 0.03


def test_module_matches_dataset(df):
    pre, _, _ = ligamx.replay(df)
    assert np.allclose(pre["TeamElo"].round(2), df["TeamElo"])
    assert np.allclose(pre["OponentElo"].round(2), df["OponentElo"])
    for col in ["H2HWinRate", "H2HLast5"]:
        assert np.allclose(pre[col], df[col], equal_nan=True)


@pytest.mark.parametrize("back", [1, 2, 3])
def test_state_before_a_matchday_matches_dataset(back):
    raw = pd.read_csv(DATASET)
    dates = pd.to_datetime(raw["Date"])
    day = sorted(dates.unique())[-back]
    state = ligamx.build_current_state(raw[dates < day])
    rows = ligamx.prepare(raw[dates == day])

    for _, r in rows.iterrows():
        team = state["team"][int(r.TeamID)]
        assert team["TeamElo"] == pytest.approx(r.TeamElo, abs=0.006)
        for col in ["TeamForm5", "TeamWinRate5", "TeamSeasonPts", "TeamHomeForm5", "TeamAwayForm5"]:
            assert team[col] == pytest.approx(r[col], nan_ok=True), col
        h2h = state["h2h"].get((int(r.TeamID), int(r.OpponentID)), {})
        for col in ["H2HWinRate", "H2HLast5"]:
            assert h2h.get(col, np.nan) == pytest.approx(r[col], nan_ok=True), col


def test_new_season_state():
    """A new tournament starts everyone at 0 season points; a new club gets the next free ID."""
    raw = pd.read_csv(DATASET)
    state = ligamx.build_current_state(raw, season=(2026, "Apertura"))
    assert all(team["TeamSeasonPts"] == 0 for team in state["team"].values())
    ids = ligamx.team_ids(raw, ["Atlante"])
    assert ids["Atlante"] == max(raw["TeamID"].max(), raw["OpponentID"].max()) + 1
    assert all(ids[name] == i for name, i in zip(raw["Equipo"], raw["TeamID"]))


# Clausura 2020 was cancelled after matchday 10 (COVID-19), so it is incomplete by design
CANCELLED = {(2020, "Clausura")}


@pytest.mark.xfail(strict=True, reason="Apertura 2014 lacks 1 match; Apertura 2025 and Clausura 2026 lack about half. "
                                       "Remove this marker once they are re-scraped.")
def test_tournaments_are_complete():
    raw = pd.read_csv(DATASET, parse_dates=["Date"])
    # the latest tournament may still be in progress, so it isn't checked
    latest = tuple(raw.sort_values("Date").iloc[-1][["Temporada", "Torneo"]])
    gaps = ligamx.missing_matches(raw)
    gaps = gaps[[(t, s) not in CANCELLED | {latest} for t, s in zip(gaps["Temporada"], gaps["Torneo"])]]
    assert gaps.empty, gaps.to_string()
