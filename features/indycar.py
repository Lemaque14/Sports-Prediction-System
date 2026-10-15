"""Up-to-date IndyCar features for predicting the next race.

Mirrors IndyCar/datasets/metrics_generator.ipynb (and the track/pit-stop
aggregates in the v25 dataset), but returns each feature's value *after* the
latest race instead of before it. tests/test_indycar_state.py checks that
replaying the history up to any race reproduces the dataset's own values for
that race.

Each lookup table holds the features at its natural key. Broader tables also
carry the narrower features under the same names, so the app's fallback
hierarchy (track -> track type -> career) reads one column name throughout.
"""

import numpy as np
import pandas as pd

BASE_ELO = 1500.0
DRIVER_K = 64
TEAM_K = 48
ENGINE_K = 32
RITMO = 5          # rolling form window and EWM centre of mass
DNF_WINDOW = 20


def compute_elo(elos, ranks, k):
    """Multi-competitor Elo update (same as computeELO in metrics_generator)."""
    n = len(elos)
    if n <= 1:
        return elos

    scores = (n - ranks) / (n - 1)

    diff = (elos.reshape(-1, 1) - elos.reshape(1, -1)) / 400
    expected = 1 / (1 + 10 ** (-diff))
    np.fill_diagonal(expected, np.nan)
    expected = np.nanmean(expected, axis=1)

    return elos + k * (scores - expected)


def _key(row_values):
    """Turn a tuple of numpy scalars into a plain int / tuple-of-ints key."""
    values = tuple(int(v) for v in row_values)
    return values[0] if len(values) == 1 else values


def _final_elo(rows, keys, rank_col, k):
    """Replay Elo over every event in order and return the post-race ratings."""
    ratings = {}
    for _, event in rows.groupby("EventID", sort=False):
        ids = [_key(v) for v in zip(*(event[c] for c in keys))]
        pre = np.array([ratings.get(i, BASE_ELO) for i in ids], dtype=float)
        post = compute_elo(pre, event[rank_col].to_numpy(dtype=float), k)
        ratings.update(zip(ids, post))
    return ratings


def _summarise(rows, keys, **aggs):
    """Group chronologically ordered rows by `keys` and apply named reducers."""
    grouped = rows.groupby(keys, sort=False)
    out = pd.DataFrame({name: grouped[col].agg(fn) for name, (col, fn) in aggs.items()})
    return {_key(k if isinstance(k, tuple) else (k,)): row for k, row in out.to_dict("index").items()}


def _last(n):
    return lambda s: s.tail(n).mean()


def _ewm(s):
    return s.ewm(com=RITMO).mean().iloc[-1]


def _add_elo(table, ratings, column):
    for key, row in table.items():
        row[column] = ratings[key]


def _copy(table, mapping):
    """Fill fallback columns from columns already in each row."""
    for row in table.values():
        for target, source in mapping.items():
            row[target] = row[source]


def prepare(df):
    """Parse dates and order rows the way the generator does."""
    df = df.copy()
    df["EventDate"] = pd.to_datetime(df["EventDate"])
    for col in ["DriverID", "TeamID", "EngineID", "TrackID", "EventTrackTypeID", "EventID"]:
        df[col] = df[col].astype(int)
    # Recompute instead of reading the CSV's 6-decimal copy: team and engine Elo
    # rank tied averages, and rounding would break those ties differently.
    df["NormalizedPositionFinish"] = (df["PositionFinish"] - 1) / (df["FieldSize"] - 1)
    return df.sort_values(["EventDate", "EventID"], kind="stable").reset_index(drop=True)


def build_current_state(df):
    """Return lookup tables of every feature as it stands after the latest race.

    Keys: driver, driver_track, driver_tracktype, team, team_track,
    team_tracktype, engine, engine_track, engine_tracktype, track, tracktype.
    """
    df = prepare(df)
    npf, dnf = "NormalizedPositionFinish", "StatusID"

    # Drivers
    driver = _summarise(df, ["DriverID"],
                        DRFAvg=(npf, _last(RITMO)), DNFRate=(dnf, _last(DNF_WINDOW)),
                        DriverRitmo=(npf, _ewm),
                        DTAvg=(npf, "mean"), TDNFRate=(dnf, "mean"))
    driver_track = _summarise(df, ["DriverID", "TrackID"], DTAvg=(npf, "mean"), TDNFRate=(dnf, "mean"))
    driver_tracktype = _summarise(df, ["DriverID", "EventTrackTypeID"], DTTAvg=(npf, "mean"), TDNFRate=(dnf, "mean"))

    _add_elo(driver, _final_elo(df, ["DriverID"], "PositionFinish", DRIVER_K), "DriverElo")
    _add_elo(driver_track, _final_elo(df, ["DriverID", "TrackID"], "PositionFinish", DRIVER_K), "DriverTElo")
    _add_elo(driver_tracktype, _final_elo(df, ["DriverID", "EventTrackTypeID"], "PositionFinish", DRIVER_K), "DriverTTElo")
    _copy(driver_tracktype, {"DTAvg": "DTTAvg", "DriverTElo": "DriverTTElo"})
    _copy(driver, {"DTTAvg": "DTAvg", "DriverTElo": "DriverElo", "DriverTTElo": "DriverElo"})

    # Teams: one row per team per race (mean finish of its cars, any DNF)
    team_rows = (
        df.groupby(["EventID", "TeamID"], as_index=False, sort=False)
        .agg(EventDate=("EventDate", "first"), TrackID=("TrackID", "first"),
             EventTrackTypeID=("EventTrackTypeID", "first"),
             teamPerf=(npf, "mean"), teamDNF=(dnf, "max"))
        .sort_values(["EventDate", "EventID", "TeamID"], kind="stable")
        .reset_index(drop=True)
    )
    team_rows["teamRank"] = team_rows.groupby("EventID", sort=False)["teamPerf"].rank(method="first")

    team = _summarise(team_rows, ["TeamID"],
                      TRP=("teamPerf", _last(RITMO)), TeamDNFRate=("teamDNF", _last(DNF_WINDOW)),
                      TeamRitmo=("teamPerf", _ewm), TTP=("teamPerf", "mean"))
    team_tracktype = _summarise(team_rows, ["TeamID", "EventTrackTypeID"], TTP=("teamPerf", "mean"))
    team_track = {key: {"TeamTElo": elo}
                  for key, elo in _final_elo(team_rows, ["TeamID", "TrackID"], "teamRank", TEAM_K).items()}
    _add_elo(team, _final_elo(team_rows, ["TeamID"], "teamRank", TEAM_K), "TeamElo")
    _copy(team, {"TeamTElo": "TeamElo"})

    # Team pit-stop habits are averaged over the team's car rows, not per race
    for table, keys in [(team, ["TeamID"]), (team_track, ["TeamID", "TrackID"])]:
        pits = _summarise(df, keys, TeamTrackAvgPitStops=("PitStops", "mean"),
                          TeamTrackAvgPittedCautionPCT=("PittedCautionPCT", "mean"))
        for key, row in table.items():
            row.update(pits[key])

    # Engines: one row per engine per race. The generator aggregates engines after
    # re-sorting cars by team and driver, which decides the "first" TrackID when an
    # EventID holds rows from two tracks (2014 Indy 500, Thermal/Road America 2025).
    engine_rows = (
        df.sort_values(["EventDate", "EventID", "TeamID", "DriverID"], kind="stable")
        .groupby(["EventID", "EngineID"], as_index=False, sort=False)
        .agg(EventDate=("EventDate", "first"), TrackID=("TrackID", "first"),
             EventTrackTypeID=("EventTrackTypeID", "first"), enginePerf=(npf, "mean"))
        .sort_values(["EventDate", "EventID", "EngineID"], kind="stable")
        .reset_index(drop=True)
    )
    engine_rows["engineRank"] = engine_rows.groupby("EventID", sort=False)["enginePerf"].rank(method="first")

    def engine_elo(keys, column):
        ratings = _final_elo(engine_rows, keys, "engineRank", ENGINE_K)
        return {key: {column: elo} for key, elo in ratings.items()}

    engine = engine_elo(["EngineID"], "EngineElo")
    engine_track = engine_elo(["EngineID", "TrackID"], "EngineTElo")
    engine_tracktype = engine_elo(["EngineID", "EventTrackTypeID"], "EngineTTElo")
    _copy(engine_tracktype, {"EngineTElo": "EngineTTElo"})
    _copy(engine, {"EngineTElo": "EngineElo", "EngineTTElo": "EngineElo"})

    # Tracks: race-level averages, plus pit stops averaged over car rows
    event_rows = df.drop_duplicates("EventID")
    track = _summarise(event_rows, ["TrackID"],
                       TrackAvgCautions=("TotalCautions", "mean"),
                       TrackAvgCautionLaps=("TotalCautionLaps", "mean"),
                       TrackAvgSpeed=("AvgRaceSpeed", "mean"),
                       TotalRaceLaps=("TotalRaceLaps", "last"),
                       TrackLength=("TrackLength", "last"))
    pits = _summarise(df, ["TrackID"], TrackAvgPitStops=("PitStops", "mean"))
    for key, row in track.items():
        row.update(pits[key])
        row["FuelWindowEstimate"] = row["TotalRaceLaps"] / row["TrackAvgPitStops"]
    tracktype = _summarise(event_rows, ["EventTrackTypeID"],
                           TrackTypeAvgCautions=("TotalCautions", "mean"),
                           TrackTypeAvgCautionLaps=("TotalCautionLaps", "mean"))

    return {
        "driver": driver, "driver_track": driver_track, "driver_tracktype": driver_tracktype,
        "team": team, "team_track": team_track, "team_tracktype": team_tracktype,
        "engine": engine, "engine_track": engine_track, "engine_tracktype": engine_tracktype,
        "track": track, "tracktype": tracktype,
    }
