# Looks up a team's FIFA ranking points as they were on a given date, so a match
# never sees a ranking from the future. Used to build the ranking-gap feature.
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


# A few teams are named differently in the ranking file than in our match data.
# Most names line up once accents/capitals are stripped; these need a manual fix.
RANKING_NAME_ALIASES = {
    "DR Congo": "Congo DR",
    "South Korea": "Korea Republic",
    "United States": "USA",
}


def _normalize(name: str) -> str:
    # Lowercase and drop accents so "Türkiye" and "Turkiye" match each other.
    return (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip()
    )


class RankingLookup:
    """Finds a team's ranking points as of a date, never using future data."""

    def __init__(self, rankings: pd.DataFrame) -> None:
        # Store each team's rankings as two sorted arrays: dates and the points.
        self._series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for team, group in rankings.groupby("team"):
            ordered = group.sort_values("date")
            self._series[team] = (
                ordered["date"].to_numpy(dtype="datetime64[ns]"),
                ordered["total_points"].to_numpy(dtype=float),
            )
        self._normalized = {_normalize(team): team for team in self._series}

    def _resolve(self, team: str) -> Optional[str]:
        # Match our team name to the name used in the ranking file.
        if team in RANKING_NAME_ALIASES:
            return RANKING_NAME_ALIASES[team]
        return self._normalized.get(_normalize(team))

    def points_as_of(self, team: str, date) -> Optional[float]:
        # Most recent ranking published strictly before this date (None if none yet).
        resolved = self._resolve(team)
        if resolved is None:
            return None
        dates, points = self._series[resolved]
        index = int(np.searchsorted(dates, np.datetime64(date), side="left"))
        return float(points[index - 1]) if index > 0 else None

    def latest_points(self, team: str) -> Optional[float]:
        # The team's newest ranking; used when predicting upcoming 2026 matches.
        resolved = self._resolve(team)
        if resolved is None:
            return None
        _, points = self._series[resolved]
        return float(points[-1]) if len(points) else None


def read_rankings(path: Union[str, Path]) -> pd.DataFrame:
    rankings = pd.read_csv(path, parse_dates=["date"])
    missing = {"team", "date", "total_points"} - set(rankings.columns)
    if missing:
        raise ValueError(f"Missing ranking columns: {sorted(missing)}")
    return rankings


def load_ranking_lookup(path: Union[str, Path]) -> RankingLookup:
    return RankingLookup(read_rankings(path))


def ranking_difference(
    lookup: Optional[RankingLookup], home_team: str, away_team: str, date
) -> float:
    """Home minus away ranking points as of the match date, or 0 if either is unknown."""
    if lookup is None:
        return 0.0
    home_points = lookup.points_as_of(home_team, date)
    away_points = lookup.points_as_of(away_team, date)
    if home_points is None or away_points is None:
        return 0.0
    return float(home_points - away_points)
