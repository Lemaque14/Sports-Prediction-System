"""UFC fight history, pre-fight features and round-by-round outcome probabilities.

Source data: the CSVs published by github.com/Greco1899/scrape_ufc_stats, which
scrapes ufcstats.com. clean_raw() turns them into one row per fighter per fight.

history_features() gives every row the fighter's record and stats from earlier
fights only. current_state() runs that same function with a placeholder "next
fight" for each fighter, so the app's inputs are built exactly like training's.

The models answer two questions: who wins, and given who wins, how the fight
ends (a finish in round 1..5, or a decision). outcome_table() multiplies them
into one table of probabilities that adds up to 1.
"""

import re

import numpy as np
import pandas as pd

ELO_START = 1500.0
ELO_K = 32
OUTCOMES = ["R1", "R2", "R3", "R4", "R5", "DEC"]

# Longest names first, so "Light Heavyweight" isn't read as "Heavyweight"
WEIGHT_LIMITS = {
    "Light Heavyweight": 205, "Heavyweight": 265, "Middleweight": 185, "Welterweight": 170,
    "Lightweight": 155, "Featherweight": 145, "Bantamweight": 135, "Flyweight": 125,
}

STAT_COLUMNS = ["KD", "SigLanded", "SigAttempted", "TotalLanded", "TotalAttempted",
                "TDLanded", "TDAttempted", "SubAtt", "Rev", "CtrlSecs"]


# ---------------------------------------------------------------- cleaning

def _landed_attempted(series):
    parts = series.astype(str).str.extract(r"(\d+) of (\d+)").astype(float)
    return parts[0], parts[1]


def _seconds(series):
    parts = series.astype(str).str.extract(r"(\d+):(\d+)").astype(float)
    return parts[0] * 60 + parts[1]


def _fight_seconds(end_round, end_time, time_format):
    """Seconds fought: full earlier rounds plus the time in the last one."""
    durations = re.findall(r"\d+", time_format.split("(", 1)[1]) if "(" in time_format else []
    minutes = [int(d) for d in durations]
    if pd.isna(end_round) or pd.isna(end_time):
        return np.nan
    return sum(minutes[: int(end_round) - 1]) * 60 + end_time


def _method(method):
    method = method.strip()
    if method.startswith("Decision"):
        return "DEC"
    if method in ("KO/TKO", "TKO - Doctor's Stoppage"):
        return "KO"
    if method == "Submission":
        return "SUB"
    return "OTHER"   # overturned, DQ, could not continue


def _weight_class(name):
    for division, limit in WEIGHT_LIMITS.items():
        if division in name:
            return division, limit
    return "Catch/Open Weight", np.nan


def _inches(series):
    feet_inches = series.astype(str).str.extract(r"(\d+)' (\d+)\"").astype(float)
    inches = feet_inches[0] * 12 + feet_inches[1]
    plain = series.astype(str).str.extract(r"^([\d.]+)\"$")[0].astype(float)
    return inches.fillna(plain)


def clean_raw(events, results, stats, tott):
    """Men's fights as one row per fighter per fight, plus a fighter table.

    Names that belong to two different fighters can't be told apart (the fight
    files carry no fighter IDs), so their fights and attributes are left out.
    """
    events = events.assign(EVENT=events["EVENT"].str.strip(),
                           Date=pd.to_datetime(events["DATE"], format="mixed"))
    results = results.assign(EVENT=results["EVENT"].str.strip(),
                             BOUT=results["BOUT"].str.strip().str.replace(r"\s+", " ", regex=True))
    stats = stats.assign(EVENT=stats["EVENT"].str.strip(),
                         BOUT=stats["BOUT"].str.strip().str.replace(r"\s+", " ", regex=True),
                         FIGHTER=stats["FIGHTER"].str.strip())

    fights = results.merge(events[["EVENT", "Date"]], on="EVENT", how="inner")
    fights = fights[~fights["WEIGHTCLASS"].str.contains("Women", na=False)].copy()
    names = fights["BOUT"].str.split(" vs. ", n=1, expand=True)
    fights["A"], fights["B"] = names[0].str.strip(), names[1].str.strip()
    outcome = fights["OUTCOME"].str.split("/", expand=True)
    fights["ResultA"], fights["ResultB"] = outcome[0], outcome[1]
    fights["FightID"] = fights["URL"].str.rstrip("/").str.split("/").str[-1]
    fights["Method"] = fights["METHOD"].map(_method)
    fights["EndRound"] = fights["ROUND"].astype(float)
    end_time = _seconds(fights["TIME"])
    fights["FightSecs"] = [_fight_seconds(r, t, f) for r, t, f in zip(fights["EndRound"], end_time, fights["TIME FORMAT"])]
    fights["ScheduledRounds"] = fights["TIME FORMAT"].str.extract(r"^(\d) Rnd \(5")[0].astype(float)
    division = fights["WEIGHTCLASS"].str.strip().map(_weight_class)
    fights["WeightClass"] = division.str[0]
    fights["WeightLbs"] = division.str[1]
    fights["Title"] = fights["WEIGHTCLASS"].str.contains("Title", na=False).astype(int)

    # Per-fight totals for each fighter, summed over rounds
    rounds = stats[stats["ROUND"].astype(str).str.startswith("Round")].copy()
    rounds["SigLanded"], rounds["SigAttempted"] = _landed_attempted(rounds["SIG.STR."])
    rounds["TotalLanded"], rounds["TotalAttempted"] = _landed_attempted(rounds["TOTAL STR."])
    rounds["TDLanded"], rounds["TDAttempted"] = _landed_attempted(rounds["TD"])
    rounds["CtrlSecs"] = _seconds(rounds["CTRL"])
    rounds = rounds.rename(columns={"SUB.ATT": "SubAtt", "REV.": "Rev"})
    totals = rounds.groupby(["EVENT", "BOUT", "FIGHTER"], as_index=False)[STAT_COLUMNS].sum(min_count=1)

    keep = ["FightID", "Date", "EVENT", "BOUT", "WeightClass", "WeightLbs", "Title",
            "ScheduledRounds", "Method", "EndRound", "FightSecs"]
    sides = []
    for me, them in [("A", "B"), ("B", "A")]:
        side = fights[keep].copy()
        side["Fighter"], side["Opponent"], side["Result"] = fights[me], fights[them], fights[f"Result{me}"]
        sides.append(side)
    long = pd.concat(sides, ignore_index=True)
    own = totals.rename(columns={"FIGHTER": "Fighter"})
    opp = totals.rename(columns={"FIGHTER": "Opponent", **{c: f"Opp{c}" for c in STAT_COLUMNS}})
    long = long.merge(own, on=["EVENT", "BOUT", "Fighter"], how="left")
    long = long.merge(opp, on=["EVENT", "BOUT", "Opponent"], how="left")

    fighters = pd.DataFrame({
        "Fighter": tott["FIGHTER"].str.strip(),
        "HeightIn": _inches(tott["HEIGHT"]),
        "ReachIn": _inches(tott["REACH"]),
        "Stance": tott["STANCE"].where(tott["STANCE"].isin(["Orthodox", "Southpaw", "Switch"])),
        "DOB": pd.to_datetime(tott["DOB"], format="%b %d, %Y", errors="coerce"),
    })
    shared = fighters["Fighter"][fighters["Fighter"].duplicated()].unique()
    fighters = fighters[~fighters["Fighter"].isin(shared)]
    unclear = long.loc[long["Fighter"].isin(shared), "FightID"].unique()
    long = long[~long["FightID"].isin(unclear)]

    long = long.drop(columns="BOUT").rename(columns={"EVENT": "Event"})
    long = long.sort_values(["Date", "FightID", "Fighter"], kind="stable").reset_index(drop=True)
    return long, fighters.sort_values("Fighter").reset_index(drop=True)


# ---------------------------------------------------------------- features

def _elo(long):
    """Each row's Elo before the fight; draws count as half a win, no contests don't count."""
    ratings, pre = {}, np.full(len(long), ELO_START)
    for _, fight in long.groupby("FightID", sort=False):
        idx = fight.index.to_numpy()
        names = fight["Fighter"].tolist()
        pre[idx] = [ratings.get(n, ELO_START) for n in names]
        if len(idx) != 2 or fight["Result"].iloc[0] not in ("W", "L", "D"):
            continue
        score = {"W": 1.0, "L": 0.0, "D": 0.5}[fight["Result"].iloc[0]]
        a, b = pre[idx[0]], pre[idx[1]]
        expected = 1 / (1 + 10 ** ((b - a) / 400))
        ratings[names[0]] = a + ELO_K * (score - expected)
        ratings[names[1]] = b - ELO_K * (score - expected)
    return pre


def _streak(results):
    """Consecutive wins (positive) or losses (negative) before each fight."""
    out, current = [], 0
    for r in results:
        out.append(current)
        if r == "W":
            current = current + 1 if current > 0 else 1
        elif r == "L":
            current = current - 1 if current < 0 else -1
        else:
            current = 0
    return out


def history_features(long, fighters):
    """Pre-fight features for every row, from that fighter's earlier fights only."""
    long = long.sort_values(["Date", "FightID", "Fighter"], kind="stable").reset_index(drop=True)
    out = long[["FightID", "Date", "Fighter", "Opponent"]].copy()
    by = long.groupby("Fighter", sort=False)

    def before(values):
        """Sum over the fighter's earlier fights (this row excluded)."""
        values = values.fillna(0)
        return values.groupby(long["Fighter"]).cumsum() - values

    result, method = long["Result"], long["Method"]
    won, lost = (result == "W").astype(float), (result == "L").astype(float)
    fought = long["Result"].notna().astype(float)          # placeholders for the next fight don't count
    has_stats = long["SigAttempted"].notna() & long["FightSecs"].notna()
    secs = long["FightSecs"].where(has_stats, 0)

    n = before(fought)
    wins, losses = before(won), before(lost)
    out["Fights"] = n
    out["WinRate"] = wins / (wins + losses)
    for name, mask in [("KOWin", won * (method == "KO")), ("SubWin", won * (method == "SUB")),
                       ("KOLoss", lost * (method == "KO")), ("SubLoss", lost * (method == "SUB")),
                       ("Decision", fought * (method == "DEC"))]:
        out[f"{name}Rate"] = before(mask) / n
    minutes = before(secs) / 60
    stat = {c: before(long[c].where(has_stats)) for c in STAT_COLUMNS}
    opp = {c: before(long[f"Opp{c}"].where(has_stats)) for c in STAT_COLUMNS}
    out["AvgFightMin"] = minutes / before(has_stats.astype(float))
    out["SigLandedPerMin"] = stat["SigLanded"] / minutes
    out["SigAbsorbedPerMin"] = opp["SigLanded"] / minutes
    out["SigAccuracy"] = stat["SigLanded"] / stat["SigAttempted"]
    out["SigDefense"] = 1 - opp["SigLanded"] / opp["SigAttempted"]
    out["TDPer15"] = stat["TDLanded"] * 15 / minutes
    out["TDAccuracy"] = stat["TDLanded"] / stat["TDAttempted"]
    out["TDDefense"] = 1 - opp["TDLanded"] / opp["TDAttempted"]
    out["SubAttPer15"] = stat["SubAtt"] * 15 / minutes
    out["KDPer15"] = stat["KD"] * 15 / minutes
    out["KDAbsorbedPer15"] = opp["KD"] * 15 / minutes
    out["CtrlShare"] = stat["CtrlSecs"] / (minutes * 60)
    out["CtrlAgainstShare"] = opp["CtrlSecs"] / (minutes * 60)
    out = out.replace([np.inf, -np.inf], np.nan)

    out["Last3WinRate"] = by["Result"].transform(
        lambda s: (s == "W").astype(float).where(s.notna()).shift(1).rolling(3, min_periods=1).mean())
    out["Streak"] = by["Result"].transform(lambda s: pd.Series(_streak(s), index=s.index))
    out["DaysSinceLast"] = (long["Date"] - by["Date"].shift(1)).dt.days
    out["Elo"] = _elo(long)

    attrs = fighters.set_index("Fighter")
    out["HeightIn"] = long["Fighter"].map(attrs["HeightIn"])
    out["ReachIn"] = long["Fighter"].map(attrs["ReachIn"])
    out["Stance"] = long["Fighter"].map(attrs["Stance"])
    out["Age"] = (long["Date"] - long["Fighter"].map(attrs["DOB"])).dt.days / 365.25
    return out


FEATURES = [
    "Fights", "WinRate", "KOWinRate", "SubWinRate", "KOLossRate", "SubLossRate", "DecisionRate",
    "AvgFightMin", "SigLandedPerMin", "SigAbsorbedPerMin", "SigAccuracy", "SigDefense", "TDPer15",
    "TDAccuracy", "TDDefense", "SubAttPer15", "KDPer15", "KDAbsorbedPer15", "CtrlShare",
    "CtrlAgainstShare", "Last3WinRate", "Streak", "DaysSinceLast", "Elo", "HeightIn", "ReachIn", "Age"]
CONTEXT = ["ScheduledRounds", "Title", "WeightLbs"]


def matchup_rows(a, b, context):
    """Model input for fighter A against fighter B: both fighters' features and A minus B.

    a, b: feature rows (same length, same order); context: ScheduledRounds, Title, WeightLbs.
    """
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    rows = pd.DataFrame(index=a.index)
    for c in FEATURES:
        rows[f"A_{c}"] = a[c].astype(float)
        rows[f"B_{c}"] = b[c].astype(float)
        rows[f"D_{c}"] = rows[f"A_{c}"] - rows[f"B_{c}"]
    rows["A_Stance"], rows["B_Stance"] = a["Stance"].fillna("Unknown"), b["Stance"].fillna("Unknown")
    context = context.reset_index(drop=True)
    for c in CONTEXT:
        rows[c] = context[c].astype(float)
    return rows


def fight_table(long, feats):
    """One row per fight from the first-listed fighter's side, with features, context and outcome."""
    side = long.reset_index(drop=True).merge(feats, on=["FightID", "Date", "Fighter", "Opponent"])
    first = side.drop_duplicates("FightID", keep="first").set_index("FightID")
    second = side[side.duplicated("FightID", keep="first")].set_index("FightID")
    ids = first.index.intersection(second.index)
    first, second = first.loc[ids], second.loc[ids]
    rows = matchup_rows(first.reset_index(), second.reset_index(), first.reset_index())
    meta = first.reset_index()[["FightID", "Date", "Event", "Fighter", "Opponent", "Result",
                                "Method", "EndRound", "ScheduledRounds", "WeightClass"]]
    meta = meta.rename(columns={"Fighter": "FighterA", "Opponent": "FighterB", "ScheduledRounds": "Rounds"})
    return pd.concat([meta, rows], axis=1)


def swap(rows):
    """The same matchups seen from the other fighter's side."""
    a = rows.filter(regex="^B_").rename(columns=lambda c: c[2:])
    b = rows.filter(regex="^A_").rename(columns=lambda c: c[2:])
    return matchup_rows(a, b, rows[CONTEXT])


def outcome_label(method, end_round):
    """How a decided fight ended: R1..R5 for a KO or submission, DEC for a decision."""
    if method == "DEC":
        return "DEC"
    if method in ("KO", "SUB") and 1 <= end_round <= 5:
        return f"R{int(end_round)}"
    return None


def winner_training_rows(table):
    """Each decided fight from both sides, so the listed order can't hint at the winner."""
    decided = table[table["Result"].isin(["W", "L"])]
    flipped = swap(decided)
    forward = decided[flipped.columns].reset_index(drop=True)
    meta = decided[["FightID", "Date"]].reset_index(drop=True)
    both = pd.concat([
        pd.concat([meta, forward], axis=1).assign(AWon=(decided["Result"] == "W").astype(int).to_numpy()),
        pd.concat([meta, flipped], axis=1).assign(AWon=(decided["Result"] == "L").astype(int).to_numpy()),
    ])
    return both.sort_values(["Date", "FightID"], kind="stable").reset_index(drop=True)


def round_training_rows(table):
    """Decided fights from the winner's side, labelled with how they ended."""
    decided = table[table["Result"].isin(["W", "L"])].copy()
    decided["Outcome"] = [outcome_label(m, r) for m, r in zip(decided["Method"], decided["EndRound"])]
    decided = decided[decided["Outcome"].notna() & decided["ScheduledRounds"].isin([3, 5])]
    winner_first = decided[decided["Result"] == "W"]
    loser_first = decided[decided["Result"] == "L"]
    swapped = swap(loser_first)
    cols = swapped.columns
    rows = pd.concat([
        pd.concat([winner_first[["FightID", "Date"]].reset_index(drop=True),
                   winner_first[cols].reset_index(drop=True)], axis=1).assign(Outcome=winner_first["Outcome"].to_numpy()),
        pd.concat([loser_first[["FightID", "Date"]].reset_index(drop=True), swapped], axis=1)
        .assign(Outcome=loser_first["Outcome"].to_numpy()),
    ])
    return rows.sort_values(["Date", "FightID"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------- live state and outcome tables

def current_state(long, fighters, as_of):
    """Every fighter's features going into their next fight, as of a date.

    Adds a placeholder fight after each fighter's last one and runs the same
    history_features() as training, so the app can't drift from the model's inputs.
    """
    long = long[long["Date"] < pd.Timestamp(as_of)]
    last = long.sort_values("Date").groupby("Fighter").tail(1)
    nxt = pd.DataFrame({
        "FightID": "next_" + last["Fighter"],
        # same date type as the history: pandas 2.1 can't concatenate second- and nanosecond-precision dates
        "Date": pd.Series(pd.Timestamp(as_of), index=last.index).astype(long["Date"].dtype),
        "Fighter": last["Fighter"], "Opponent": np.nan,
    })
    feats = history_features(pd.concat([long, nxt], ignore_index=True), fighters)
    state = feats[feats["FightID"].str.startswith("next_")].set_index("Fighter")
    state["LastFight"] = last.set_index("Fighter")["Date"]
    state["LastWeightClass"] = last.set_index("Fighter")["WeightClass"]
    return state.drop(columns=["FightID", "Opponent"])


def outcome_table(p_a_wins, how_a, how_b, rounds):
    """Probability of each fighter winning in each round or by decision.

    p_a_wins: P(A wins); how_a / how_b: probabilities over OUTCOMES given that
    fighter wins. Rounds beyond the scheduled number are zeroed out.
    """
    allowed = np.array([o == "DEC" or int(o[1]) <= rounds for o in OUTCOMES], dtype=float)
    how_a = np.asarray(how_a, dtype=float) * allowed
    how_b = np.asarray(how_b, dtype=float) * allowed
    how_a, how_b = how_a / how_a.sum(), how_b / how_b.sum()
    table = pd.DataFrame([p_a_wins * how_a, (1 - p_a_wins) * how_b], columns=OUTCOMES, index=["A", "B"])
    return table[[o for o, ok in zip(OUTCOMES, allowed) if ok]]


def symmetric_win_probability(p_ab, p_ba):
    """P(A wins) from the model's answer in both orders, so swapping fighters swaps the result."""
    return (np.asarray(p_ab) + 1 - np.asarray(p_ba)) / 2
