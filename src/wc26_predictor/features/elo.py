# Elo ratings: a running strength score for every team that goes up after a win
# and down after a loss. How much it moves depends on how big the match was and
# how heavily the team won.
from __future__ import annotations

import math

MAJOR_TOURNAMENTS = (
    "uefa euro",
    "copa américa",
    "copa america",
    "african cup of nations",
    "afc asian cup",
    "gold cup",
)


def match_weight(tournament: str) -> float:
    # Big games should move the ratings more than friendlies do.
    name = str(tournament).lower()
    if "friendly" in name:
        return 20.0
    if "qualification" in name or "qualifier" in name:
        return 40.0
    if name == "fifa world cup":
        return 60.0
    if name in MAJOR_TOURNAMENTS:
        return 50.0
    return 30.0


def margin_multiplier(home_score: int, away_score: int) -> float:
    # A bigger winning margin is stronger evidence, so it moves the rating more.
    margin = abs(home_score - away_score)
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return 1.75 + (margin - 3) / 8


class EloRatings:
    def __init__(
        self,
        default_rating: float = 1500.0,
        k_factor: float = 32.0,
        home_advantage: float = 100.0,
    ) -> None:
        self.default_rating = default_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: dict[str, float] = {}

    def get(self, team: str) -> float:
        # New teams we have not seen yet start at the average rating (1500).
        return self.ratings.get(team, self.default_rating)

    def expected_score(self, team_a: str, team_b: str, advantage: float = 0.0) -> float:
        # Win probability for team A based on the rating gap (standard Elo formula).
        rating_a = self.get(team_a) + advantage
        rating_b = self.get(team_b)
        return 1 / (1 + math.pow(10, (rating_b - rating_a) / 400))

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        neutral: bool = True,
        tournament: str | None = None,
    ) -> None:
        # Actual result for the home team: 1 = win, 0.5 = draw, 0 = loss.
        if home_score > away_score:
            actual_home = 1.0
        elif home_score == away_score:
            actual_home = 0.5
        else:
            actual_home = 0.0

        # How much to move the ratings: bigger for important games and big wins.
        k = self.k_factor if tournament is None else match_weight(tournament)
        k *= margin_multiplier(home_score, away_score)
        # Give the home team a rating boost when it is not a neutral venue.
        advantage = 0.0 if neutral else self.home_advantage
        expected_home = self.expected_score(home_team, away_team, advantage)
        expected_away = 1 - expected_home
        actual_away = 1 - actual_home

        # Move each rating by (what happened - what we expected).
        self.ratings[home_team] = self.get(home_team) + k * (actual_home - expected_home)
        self.ratings[away_team] = self.get(away_team) + k * (actual_away - expected_away)
