from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


TOURNAMENT_START_DATE = pd.Timestamp("2026-06-11")

MATCH_COLUMNS = {
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "neutral",
}

SQUAD_BASE_COLUMNS = {
    "team",
    "player",
    "position",
    "caps",
    "club_strength",
    "attacking_value",
    "defensive_value",
}

QUALIFIED_TEAM_COLUMNS = {"confederation", "team", "qualification_path"}

SOURCE_TEAM_ALIASES = {
    "Cape Verde": "Cabo Verde",
    "Curaçao": "Curacao",
    "Czech Republic": "Czechia",
    "Iran": "IR Iran",
    "Ivory Coast": "Cote d'Ivoire",
    "Turkey": "Turkiye",
}


def canonical_team_name(team: str) -> str:
    # Use one consistent spelling per team (e.g. always "Turkiye", never "Turkey").
    return SOURCE_TEAM_ALIASES.get(team, team)


def _age_on_tournament_start(birth_dates: pd.Series) -> pd.Series:
    # Each player's age on the day the tournament kicks off.
    parsed = pd.to_datetime(birth_dates, errors="coerce", format="mixed")
    ages = TOURNAMENT_START_DATE.year - parsed.dt.year
    # Subtract a year if their birthday has not happened yet by kickoff.
    had_birthday = (parsed.dt.month < TOURNAMENT_START_DATE.month) | (
        (parsed.dt.month == TOURNAMENT_START_DATE.month)
        & (parsed.dt.day <= TOURNAMENT_START_DATE.day)
    )
    return (ages - (~had_birthday).astype(int)).astype("Int64")


def read_matches(path: Union[str, Path]) -> pd.DataFrame:
    # Load the match history and tidy it up (sort by date, fix team names).
    matches = pd.read_csv(path)
    missing = MATCH_COLUMNS - set(matches.columns)
    if missing:
        raise ValueError(f"Missing match columns: {sorted(missing)}")

    matches["date"] = pd.to_datetime(matches["date"], format="%Y-%m-%d")
    matches = matches.sort_values("date").reset_index(drop=True)
    matches["home_team"] = matches["home_team"].map(canonical_team_name)
    matches["away_team"] = matches["away_team"].map(canonical_team_name)
    matches["neutral"] = matches["neutral"].astype(bool)
    return matches


def read_squads(path: Union[str, Path]) -> pd.DataFrame:
    squads = pd.read_csv(path)
    squads = squads.rename(
        columns={
            "birthday": "birth_date",
            "club strength": "club_strength",
        }
    )
    missing = SQUAD_BASE_COLUMNS - set(squads.columns)
    if missing:
        raise ValueError(f"Missing squad columns: {sorted(missing)}")
    if "birth_date" in squads.columns:
        squads["age"] = _age_on_tournament_start(squads["birth_date"])
    elif "age" not in squads.columns:
        raise ValueError("Squads must include either birth_date or age")
    if squads["age"].isna().any():
        bad_rows = squads[squads["age"].isna()][["team", "player"]].to_dict(
            orient="records"
        )
        raise ValueError(f"Could not calculate age for squad rows: {bad_rows}")
    squads["age"] = squads["age"].astype(int)
    return squads


def read_qualified_teams(path: Union[str, Path]) -> pd.DataFrame:
    teams = pd.read_csv(path)
    missing = QUALIFIED_TEAM_COLUMNS - set(teams.columns)
    if missing:
        raise ValueError(f"Missing qualified-team columns: {sorted(missing)}")
    return teams.sort_values(["confederation", "team"]).reset_index(drop=True)


def aggregate_squad_features(squads: pd.DataFrame) -> pd.DataFrame:
    # Roll a team's individual players up into one row of team-level squad stats.
    numeric_cols = [
        "age",
        "caps",
        "club_strength",
        "attacking_value",
        "defensive_value",
    ]
    for col in numeric_cols:
        if col not in squads:
            raise ValueError(f"Expected numeric squad column {col}")

    return (
        squads.groupby("team", as_index=False)
        .agg(
            squad_avg_age=("age", "mean"),
            squad_avg_caps=("caps", "mean"),
            squad_total_caps=("caps", "sum"),
            squad_club_strength=("club_strength", "mean"),
            squad_attack=("attacking_value", "sum"),
            squad_defense=("defensive_value", "sum"),
            squad_size=("player", "count"),
        )
        .fillna(0)
    )
