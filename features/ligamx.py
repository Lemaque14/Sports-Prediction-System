"""Liga MX Elo, head-to-head and form features.

Mirrors LigaMX/datasets/LigaMX_metrics.ipynb. Each match appears twice in the
dataset (once from each team's side); both rows must get the values from
*before* the match. tests/test_ligamx_state.py checks this module against the
dataset the notebook writes.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

K = 32
START_ELO = 1500
HOME_ADVANTAGE = 50
WINDOW = 5
POINTS = {"W": 3, "D": 1, "L": 0}
FLIP = {"W": "L", "L": "W", "D": "D"}


def prepare(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "Equipo"], kind="stable").reset_index(drop=True)
    df["match_id"] = [(min(t, o), max(t, o), d) for t, o, d in zip(df["Equipo"], df["Opponent"], df["Date"])]
    return df


def _update_elo(ratings, team, opponent, venue, result):
    team_elo, opp_elo = ratings[team], ratings[opponent]
    if venue == "Home":
        team_elo += HOME_ADVANTAGE
    else:
        opp_elo += HOME_ADVANTAGE
    exp_team = 1 / (1 + 10 ** ((opp_elo - team_elo) / 400))
    actual = {"W": 1.0, "L": 0.0}.get(result, 0.5)
    ratings[team] += K * (actual - exp_team)
    ratings[opponent] += K * ((1 - actual) - (1 - exp_team))


def replay(df):
    """Walk every match once, in date order.

    Returns per-row pre-match values (TeamElo, OponentElo, H2HWinRate, H2HLast5)
    and the final Elo / head-to-head state after the last match.
    """
    ratings = defaultdict(lambda: float(START_ELO))
    h2h_wins, h2h_matches, h2h_last = defaultdict(int), defaultdict(int), defaultdict(list)
    pre = {}

    for _, match in df.groupby("match_id", sort=False):
        # Record what both rows knew before kick-off, then update once.
        for idx, row in match.iterrows():
            pair = (row["Equipo"], row["Opponent"])
            played = h2h_matches[pair]
            last5 = h2h_last[pair][-WINDOW:]
            pre[idx] = (ratings[row["Equipo"]], ratings[row["Opponent"]],
                        h2h_wins[pair] / played if played else np.nan,
                        sum(last5) if last5 else np.nan)

        row = match.iloc[0]
        team, opponent, result = row["Equipo"], row["Opponent"], row["Result"]
        _update_elo(ratings, team, opponent, row["Venue"], result)
        h2h_matches[(team, opponent)] += 1
        h2h_matches[(opponent, team)] += 1
        if result == "W":
            h2h_wins[(team, opponent)] += 1
        elif result == "L":
            h2h_wins[(opponent, team)] += 1
        h2h_last[(team, opponent)].append(POINTS[result])
        h2h_last[(opponent, team)].append(POINTS[FLIP[result]])

    pre = pd.DataFrame.from_dict(pre, orient="index", columns=["TeamElo", "OponentElo", "H2HWinRate", "H2HLast5"])
    h2h = {
        pair: {"H2HWinRate": h2h_wins[pair] / n, "H2HLast5": sum(h2h_last[pair][-WINDOW:])}
        for pair, n in h2h_matches.items() if n
    }
    return pre.sort_index(), dict(ratings), h2h


def _form(rows):
    """Form features a team carries into its next match (rows: its own, in order)."""
    last = rows.tail(WINDOW)
    home = last.loc[last["Venue"] == "Home", "Points"]
    away = last.loc[last["Venue"] == "Away", "Points"]
    season = rows[(rows["Temporada"] == rows["Temporada"].iloc[-1]) & (rows["Torneo"] == rows["Torneo"].iloc[-1])]
    return {
        "TeamForm5": last["Points"].sum(),
        "TeamWinRate5": (last["Result"] == "W").mean(),
        "TeamSeasonPts": season["Points"].sum(),
        "TeamHomeForm5": home.sum() if len(home) else np.nan,
        "TeamAwayForm5": away.sum() if len(away) else np.nan,
    }


def build_current_state(df):
    """Features as they stand after the latest match, keyed like the app's IDs.

    Returns {"team": {TeamID: {...}}, "h2h": {(TeamID, OpponentID): {...}}}.
    """
    df = prepare(df)
    df["Points"] = df["Result"].map(POINTS)
    _, ratings, h2h = replay(df)

    team_ids = dict(zip(df["Equipo"], df["TeamID"]))
    opp_ids = dict(zip(df["Opponent"], df["OpponentID"]))

    team = {}
    for name, rows in df.groupby("Equipo", sort=False):
        team[int(team_ids[name])] = {"TeamElo": ratings[name], **_form(rows)}

    pairs = {
        (int(team_ids[a]), int(opp_ids[b])): values
        for (a, b), values in h2h.items() if a in team_ids and b in opp_ids
    }
    return {"team": team, "h2h": pairs}
