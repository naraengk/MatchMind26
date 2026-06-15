from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

import numpy as np
import pandas as pd

from wc26_predictor.data.loading import aggregate_squad_features
from wc26_predictor.features.elo import EloRatings
from wc26_predictor.features.rankings import RankingLookup, ranking_difference


FEATURE_COLUMNS = [
    "elo_diff",
    "abs_elo_diff",
    "ranking_diff",
    "home_recent_points",
    "away_recent_points",
    "recent_points_diff",
    "abs_recent_points_diff",
    "home_points_last_5",
    "away_points_last_5",
    "points_last_5_diff",
    "home_points_last_10",
    "away_points_last_10",
    "points_last_10_diff",
    "home_form_momentum",
    "away_form_momentum",
    "home_recent_goal_diff",
    "away_recent_goal_diff",
    "recent_goal_diff_delta",
    "abs_recent_goal_diff_delta",
    "home_goal_diff_last_5",
    "away_goal_diff_last_5",
    "goal_diff_last_5_delta",
    "home_goal_diff_last_10",
    "away_goal_diff_last_10",
    "goal_diff_last_10_delta",
    "home_recent_draw_rate",
    "away_recent_draw_rate",
    "draw_rate_average",
    "draw_rate_diff",
    "home_draw_rate_last_5",
    "away_draw_rate_last_5",
    "draw_rate_last_5_average",
    "home_draw_rate_last_10",
    "away_draw_rate_last_10",
    "draw_rate_last_10_average",
    "home_recent_goals_for",
    "away_recent_goals_for",
    "home_recent_goals_against",
    "away_recent_goals_against",
    "home_goals_for_last_5",
    "away_goals_for_last_5",
    "home_goals_against_last_5",
    "away_goals_against_last_5",
    "home_goals_for_last_10",
    "away_goals_for_last_10",
    "home_goals_against_last_10",
    "away_goals_against_last_10",
    "expected_goal_total_proxy",
    "home_low_scoring_rate",
    "away_low_scoring_rate",
    "low_scoring_average",
    "home_clean_sheet_rate",
    "away_clean_sheet_rate",
    "neutral",
    "is_world_cup",
    "is_friendly",
    "is_qualifier",
    "is_major_tournament",
    "squad_avg_age_diff",
    "squad_avg_caps_diff",
    "squad_total_caps_diff",
    "squad_club_strength_diff",
    "squad_attack_diff",
    "squad_defense_diff",
]

HISTORICAL_FEATURE_COLUMNS = [
    column for column in FEATURE_COLUMNS if not column.startswith("squad_")
]


def match_outcome(home_score: int, away_score: int) -> str:
    # The label the model learns to predict.
    if home_score > away_score:
        return "home_win"
    if home_score < away_score:
        return "away_win"
    return "draw"


def _points_for(score_for: int, score_against: int) -> int:
    # League points for one team: 3 for a win, 1 for a draw, 0 for a loss.
    if score_for > score_against:
        return 3
    if score_for == score_against:
        return 1
    return 0


def _recent_average(history: deque[float]) -> float:
    # Average of a team's recent values (last 10 games), or 0 if it has no history.
    return float(np.mean(history)) if history else 0.0


def _window_average(history: deque[float], size: int) -> float:
    # Average over just the last `size` games (e.g. last 5).
    values = list(history)[-size:]
    return float(np.mean(values)) if values else 0.0


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _competition_features(tournament: str) -> dict[str, int]:
    # Yes/no flags telling the model what kind of match this is.
    name = str(tournament).lower()
    is_world_cup = name == "fifa world cup"
    is_friendly = "friendly" in name
    is_qualifier = "qualification" in name or "qualifier" in name
    major_names = (
        "fifa world cup",
        "uefa euro",
        "copa américa",
        "copa america",
        "african cup of nations",
        "afc asian cup",
        "gold cup",
    )
    is_major_tournament = not is_qualifier and any(name == value for value in major_names)
    return {
        "is_world_cup": int(is_world_cup),
        "is_friendly": int(is_friendly),
        "is_qualifier": int(is_qualifier),
        "is_major_tournament": int(is_major_tournament),
    }


def _squad_lookup(squads: Optional[pd.DataFrame]) -> dict[str, dict[str, float]]:
    # Pre-compute each team's squad totals (attack, defense, caps, ...) once.
    if squads is None or squads.empty:
        return {}
    aggregated = aggregate_squad_features(squads)
    return aggregated.set_index("team").to_dict(orient="index")


def _squad_diff(
    lookup: dict[str, dict[str, float]], home_team: str, away_team: str, field: str
) -> float:
    # Home team's value minus the away team's for one squad stat.
    home_value = lookup.get(home_team, {}).get(field, 0.0)
    away_value = lookup.get(away_team, {}).get(field, 0.0)
    return float(home_value - away_value)


def build_training_frame(
    matches: pd.DataFrame,
    squads: Optional[pd.DataFrame] = None,
    rankings: Optional[RankingLookup] = None,
) -> pd.DataFrame:
    """Turn the match list into one row of features per match.

    Matches are walked through in date order. For each one we build the features
    from what was known *before* kickoff, then update the running stats with the
    result afterwards. That ordering is what keeps future data from leaking in.
    """
    elo = EloRatings()
    # Rolling memory of each team's last 10 games for the recent-form features.
    recent_points: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=10))
    recent_goal_diff: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=10))
    recent_draws: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=10))
    recent_goals_for: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=10))
    recent_goals_against: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=10))
    recent_low_scoring: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=10))
    recent_clean_sheets: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=10))
    squad_features = _squad_lookup(squads)
    rows: list[dict[str, object]] = []

    for match in matches.itertuples(index=False):
        # Read each team's form so far (these stats hold only earlier games).
        home_team = match.home_team
        away_team = match.away_team
        home_recent_points = _recent_average(recent_points[home_team])
        away_recent_points = _recent_average(recent_points[away_team])
        home_recent_gd = _recent_average(recent_goal_diff[home_team])
        away_recent_gd = _recent_average(recent_goal_diff[away_team])
        home_draw_rate = _recent_average(recent_draws[home_team])
        away_draw_rate = _recent_average(recent_draws[away_team])
        home_goals_for = _recent_average(recent_goals_for[home_team])
        away_goals_for = _recent_average(recent_goals_for[away_team])
        home_goals_against = _recent_average(recent_goals_against[home_team])
        away_goals_against = _recent_average(recent_goals_against[away_team])
        home_low_scoring = _recent_average(recent_low_scoring[home_team])
        away_low_scoring = _recent_average(recent_low_scoring[away_team])
        home_clean_sheet = _recent_average(recent_clean_sheets[home_team])
        away_clean_sheet = _recent_average(recent_clean_sheets[away_team])
        home_points_5 = _window_average(recent_points[home_team], 5)
        away_points_5 = _window_average(recent_points[away_team], 5)
        home_points_10 = _window_average(recent_points[home_team], 10)
        away_points_10 = _window_average(recent_points[away_team], 10)
        home_goal_diff_5 = _window_average(recent_goal_diff[home_team], 5)
        away_goal_diff_5 = _window_average(recent_goal_diff[away_team], 5)
        home_goal_diff_10 = _window_average(recent_goal_diff[home_team], 10)
        away_goal_diff_10 = _window_average(recent_goal_diff[away_team], 10)
        home_draw_rate_5 = _window_average(recent_draws[home_team], 5)
        away_draw_rate_5 = _window_average(recent_draws[away_team], 5)
        home_draw_rate_10 = _window_average(recent_draws[home_team], 10)
        away_draw_rate_10 = _window_average(recent_draws[away_team], 10)
        home_goals_for_5 = _window_average(recent_goals_for[home_team], 5)
        away_goals_for_5 = _window_average(recent_goals_for[away_team], 5)
        home_goals_against_5 = _window_average(recent_goals_against[home_team], 5)
        away_goals_against_5 = _window_average(recent_goals_against[away_team], 5)
        home_goals_for_10 = _window_average(recent_goals_for[home_team], 10)
        away_goals_for_10 = _window_average(recent_goals_for[away_team], 10)
        home_goals_against_10 = _window_average(recent_goals_against[home_team], 10)
        away_goals_against_10 = _window_average(recent_goals_against[away_team], 10)
        elo_diff = elo.get(home_team) - elo.get(away_team)
        recent_points_delta = home_recent_points - away_recent_points
        recent_goal_diff_delta = home_recent_gd - away_recent_gd
        ranking_diff = ranking_difference(rankings, home_team, away_team, match.date)

        row = {
            "date": match.date,
            "home_team": home_team,
            "away_team": away_team,
            "elo_diff": elo_diff,
            "abs_elo_diff": abs(elo_diff),
            "ranking_diff": ranking_diff,
            "home_recent_points": home_recent_points,
            "away_recent_points": away_recent_points,
            "recent_points_diff": recent_points_delta,
            "abs_recent_points_diff": abs(recent_points_delta),
            "home_points_last_5": home_points_5,
            "away_points_last_5": away_points_5,
            "points_last_5_diff": home_points_5 - away_points_5,
            "home_points_last_10": home_points_10,
            "away_points_last_10": away_points_10,
            "points_last_10_diff": home_points_10 - away_points_10,
            "home_form_momentum": home_points_5 - home_points_10,
            "away_form_momentum": away_points_5 - away_points_10,
            "home_recent_goal_diff": home_recent_gd,
            "away_recent_goal_diff": away_recent_gd,
            "recent_goal_diff_delta": recent_goal_diff_delta,
            "abs_recent_goal_diff_delta": abs(recent_goal_diff_delta),
            "home_goal_diff_last_5": home_goal_diff_5,
            "away_goal_diff_last_5": away_goal_diff_5,
            "goal_diff_last_5_delta": home_goal_diff_5 - away_goal_diff_5,
            "home_goal_diff_last_10": home_goal_diff_10,
            "away_goal_diff_last_10": away_goal_diff_10,
            "goal_diff_last_10_delta": home_goal_diff_10 - away_goal_diff_10,
            "home_recent_draw_rate": home_draw_rate,
            "away_recent_draw_rate": away_draw_rate,
            "draw_rate_average": (home_draw_rate + away_draw_rate) / 2,
            "draw_rate_diff": abs(home_draw_rate - away_draw_rate),
            "home_draw_rate_last_5": home_draw_rate_5,
            "away_draw_rate_last_5": away_draw_rate_5,
            "draw_rate_last_5_average": (home_draw_rate_5 + away_draw_rate_5) / 2,
            "home_draw_rate_last_10": home_draw_rate_10,
            "away_draw_rate_last_10": away_draw_rate_10,
            "draw_rate_last_10_average": (home_draw_rate_10 + away_draw_rate_10) / 2,
            "home_recent_goals_for": home_goals_for,
            "away_recent_goals_for": away_goals_for,
            "home_recent_goals_against": home_goals_against,
            "away_recent_goals_against": away_goals_against,
            "home_goals_for_last_5": home_goals_for_5,
            "away_goals_for_last_5": away_goals_for_5,
            "home_goals_against_last_5": home_goals_against_5,
            "away_goals_against_last_5": away_goals_against_5,
            "home_goals_for_last_10": home_goals_for_10,
            "away_goals_for_last_10": away_goals_for_10,
            "home_goals_against_last_10": home_goals_against_10,
            "away_goals_against_last_10": away_goals_against_10,
            "expected_goal_total_proxy": (
                home_goals_for + away_goals_for + home_goals_against + away_goals_against
            )
            / 2,
            "home_low_scoring_rate": home_low_scoring,
            "away_low_scoring_rate": away_low_scoring,
            "low_scoring_average": (home_low_scoring + away_low_scoring) / 2,
            "home_clean_sheet_rate": home_clean_sheet,
            "away_clean_sheet_rate": away_clean_sheet,
            "neutral": int(match.neutral),
            **_competition_features(match.tournament),
            "squad_avg_age_diff": _squad_diff(squad_features, home_team, away_team, "squad_avg_age"),
            "squad_avg_caps_diff": _squad_diff(squad_features, home_team, away_team, "squad_avg_caps"),
            "squad_total_caps_diff": _squad_diff(squad_features, home_team, away_team, "squad_total_caps"),
            "squad_club_strength_diff": _squad_diff(
                squad_features, home_team, away_team, "squad_club_strength"
            ),
            "squad_attack_diff": _squad_diff(squad_features, home_team, away_team, "squad_attack"),
            "squad_defense_diff": _squad_diff(squad_features, home_team, away_team, "squad_defense"),
            "home_goals": int(match.home_score),
            "away_goals": int(match.away_score),
            "target": match_outcome(match.home_score, match.away_score),
        }
        rows.append(row)

        # Now that the row is saved, fold this match's result into the running
        # stats so the *next* game sees it. Doing this last is what avoids leakage.
        recent_points[home_team].append(_points_for(match.home_score, match.away_score))
        recent_points[away_team].append(_points_for(match.away_score, match.home_score))
        recent_goal_diff[home_team].append(match.home_score - match.away_score)
        recent_goal_diff[away_team].append(match.away_score - match.home_score)
        is_draw = int(match.home_score == match.away_score)
        recent_draws[home_team].append(is_draw)
        recent_draws[away_team].append(is_draw)
        recent_goals_for[home_team].append(match.home_score)
        recent_goals_for[away_team].append(match.away_score)
        recent_goals_against[home_team].append(match.away_score)
        recent_goals_against[away_team].append(match.home_score)
        low_scoring = int(match.home_score + match.away_score <= 2)
        recent_low_scoring[home_team].append(low_scoring)
        recent_low_scoring[away_team].append(low_scoring)
        recent_clean_sheets[home_team].append(int(match.away_score == 0))
        recent_clean_sheets[away_team].append(int(match.home_score == 0))
        elo.update(
            home_team,
            away_team,
            match.home_score,
            match.away_score,
            neutral=bool(match.neutral),
            tournament=str(match.tournament),
        )

    return pd.DataFrame(rows)


def build_team_profiles(
    matches: pd.DataFrame,
    squads: Optional[pd.DataFrame] = None,
    team_pool: Optional[list[str]] = None,
    rankings: Optional[RankingLookup] = None,
) -> dict[str, dict]:
    """Snapshot each team's current form, Elo, squad and ranking.

    These snapshots are saved with the model and used to predict the 2026 games,
    since those fixtures have not been played yet and have no match history.
    """
    training_frame = build_training_frame(matches, squads)
    if team_pool is None:
        squad_teams = set(squads["team"]) if squads is not None else set()
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]) | squad_teams)
    else:
        teams = sorted(team_pool)

    latest_rows = {}
    for team in teams:
        team_rows = training_frame[
            (training_frame["home_team"] == team) | (training_frame["away_team"] == team)
        ]
        latest_rows[team] = team_rows.tail(5)

    elo = EloRatings()
    for match in matches.itertuples(index=False):
        elo.update(
            match.home_team,
            match.away_team,
            match.home_score,
            match.away_score,
            neutral=bool(match.neutral),
            tournament=str(match.tournament),
        )

    squad_features = _squad_lookup(squads)
    profiles = {}
    for team, rows in latest_rows.items():
        home_rows = rows[rows["home_team"] == team]
        away_rows = rows[rows["away_team"] == team]
        recent_points_values = list(home_rows["home_recent_points"]) + list(
            away_rows["away_recent_points"]
        )
        recent_gd_values = list(home_rows["home_recent_goal_diff"]) + list(
            away_rows["away_recent_goal_diff"]
        )
        def profile_values(home_field: str, away_field: str) -> list[float]:
            return list(home_rows[home_field]) + list(away_rows[away_field])

        latest_ranking = rankings.latest_points(team) if rankings is not None else None
        profiles[team] = {
            "elo": elo.get(team),
            "fifa_points": float(latest_ranking) if latest_ranking is not None else 0.0,
            "recent_points": float(np.mean(recent_points_values)) if recent_points_values else 0.0,
            "recent_goal_diff": float(np.mean(recent_gd_values)) if recent_gd_values else 0.0,
            "points_last_5": _mean_or_zero(
                profile_values("home_points_last_5", "away_points_last_5")
            ),
            "points_last_10": _mean_or_zero(
                profile_values("home_points_last_10", "away_points_last_10")
            ),
            "form_momentum": _mean_or_zero(
                profile_values("home_form_momentum", "away_form_momentum")
            ),
            "goal_diff_last_5": _mean_or_zero(
                profile_values("home_goal_diff_last_5", "away_goal_diff_last_5")
            ),
            "goal_diff_last_10": _mean_or_zero(
                profile_values("home_goal_diff_last_10", "away_goal_diff_last_10")
            ),
            "recent_draw_rate": _mean_or_zero(
                profile_values("home_recent_draw_rate", "away_recent_draw_rate")
            ),
            "draw_rate_last_5": _mean_or_zero(
                profile_values("home_draw_rate_last_5", "away_draw_rate_last_5")
            ),
            "draw_rate_last_10": _mean_or_zero(
                profile_values("home_draw_rate_last_10", "away_draw_rate_last_10")
            ),
            "recent_goals_for": _mean_or_zero(
                profile_values("home_recent_goals_for", "away_recent_goals_for")
            ),
            "recent_goals_against": _mean_or_zero(
                profile_values("home_recent_goals_against", "away_recent_goals_against")
            ),
            "goals_for_last_5": _mean_or_zero(
                profile_values("home_goals_for_last_5", "away_goals_for_last_5")
            ),
            "goals_against_last_5": _mean_or_zero(
                profile_values("home_goals_against_last_5", "away_goals_against_last_5")
            ),
            "goals_for_last_10": _mean_or_zero(
                profile_values("home_goals_for_last_10", "away_goals_for_last_10")
            ),
            "goals_against_last_10": _mean_or_zero(
                profile_values("home_goals_against_last_10", "away_goals_against_last_10")
            ),
            "low_scoring_rate": _mean_or_zero(
                profile_values("home_low_scoring_rate", "away_low_scoring_rate")
            ),
            "clean_sheet_rate": _mean_or_zero(
                profile_values("home_clean_sheet_rate", "away_clean_sheet_rate")
            ),
            **squad_features.get(team, {}),
        }
    return profiles


def build_prediction_row(
    home_team: str,
    away_team: str,
    team_profiles: dict[str, dict],
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
) -> pd.DataFrame:
    """Build one feature row for an upcoming match from the two team snapshots.

    Same columns as the training rows, so the trained model can score it directly.
    """
    home = team_profiles.get(home_team)
    away = team_profiles.get(away_team)
    if home is None or away is None:
        missing = [team for team, profile in [(home_team, home), (away_team, away)] if profile is None]
        raise ValueError(f"Missing team profiles for: {', '.join(missing)}")

    def diff(field: str) -> float:
        return float(home.get(field, 0.0) - away.get(field, 0.0))

    elo_diff = diff("elo")
    home_points = float(home.get("fifa_points", 0.0))
    away_points = float(away.get("fifa_points", 0.0))
    ranking_diff = home_points - away_points if home_points > 0 and away_points > 0 else 0.0
    recent_points_diff = diff("recent_points")
    recent_goal_diff_delta = diff("recent_goal_diff")
    home_draw_rate = float(home.get("recent_draw_rate", 0.0))
    away_draw_rate = float(away.get("recent_draw_rate", 0.0))
    home_goals_for = float(home.get("recent_goals_for", 0.0))
    away_goals_for = float(away.get("recent_goals_for", 0.0))
    home_goals_against = float(home.get("recent_goals_against", 0.0))
    away_goals_against = float(away.get("recent_goals_against", 0.0))
    home_low_scoring = float(home.get("low_scoring_rate", 0.0))
    away_low_scoring = float(away.get("low_scoring_rate", 0.0))
    home_points_5 = float(home.get("points_last_5", 0.0))
    away_points_5 = float(away.get("points_last_5", 0.0))
    home_points_10 = float(home.get("points_last_10", 0.0))
    away_points_10 = float(away.get("points_last_10", 0.0))
    home_goal_diff_5 = float(home.get("goal_diff_last_5", 0.0))
    away_goal_diff_5 = float(away.get("goal_diff_last_5", 0.0))
    home_goal_diff_10 = float(home.get("goal_diff_last_10", 0.0))
    away_goal_diff_10 = float(away.get("goal_diff_last_10", 0.0))
    home_draw_rate_5 = float(home.get("draw_rate_last_5", 0.0))
    away_draw_rate_5 = float(away.get("draw_rate_last_5", 0.0))
    home_draw_rate_10 = float(home.get("draw_rate_last_10", 0.0))
    away_draw_rate_10 = float(away.get("draw_rate_last_10", 0.0))
    row = {
        "elo_diff": elo_diff,
        "abs_elo_diff": abs(elo_diff),
        "ranking_diff": ranking_diff,
        "home_recent_points": float(home.get("recent_points", 0.0)),
        "away_recent_points": float(away.get("recent_points", 0.0)),
        "recent_points_diff": recent_points_diff,
        "abs_recent_points_diff": abs(recent_points_diff),
        "home_points_last_5": home_points_5,
        "away_points_last_5": away_points_5,
        "points_last_5_diff": home_points_5 - away_points_5,
        "home_points_last_10": home_points_10,
        "away_points_last_10": away_points_10,
        "points_last_10_diff": home_points_10 - away_points_10,
        "home_form_momentum": float(home.get("form_momentum", 0.0)),
        "away_form_momentum": float(away.get("form_momentum", 0.0)),
        "home_recent_goal_diff": float(home.get("recent_goal_diff", 0.0)),
        "away_recent_goal_diff": float(away.get("recent_goal_diff", 0.0)),
        "recent_goal_diff_delta": recent_goal_diff_delta,
        "abs_recent_goal_diff_delta": abs(recent_goal_diff_delta),
        "home_goal_diff_last_5": home_goal_diff_5,
        "away_goal_diff_last_5": away_goal_diff_5,
        "goal_diff_last_5_delta": home_goal_diff_5 - away_goal_diff_5,
        "home_goal_diff_last_10": home_goal_diff_10,
        "away_goal_diff_last_10": away_goal_diff_10,
        "goal_diff_last_10_delta": home_goal_diff_10 - away_goal_diff_10,
        "home_recent_draw_rate": home_draw_rate,
        "away_recent_draw_rate": away_draw_rate,
        "draw_rate_average": (home_draw_rate + away_draw_rate) / 2,
        "draw_rate_diff": abs(home_draw_rate - away_draw_rate),
        "home_draw_rate_last_5": home_draw_rate_5,
        "away_draw_rate_last_5": away_draw_rate_5,
        "draw_rate_last_5_average": (home_draw_rate_5 + away_draw_rate_5) / 2,
        "home_draw_rate_last_10": home_draw_rate_10,
        "away_draw_rate_last_10": away_draw_rate_10,
        "draw_rate_last_10_average": (home_draw_rate_10 + away_draw_rate_10) / 2,
        "home_recent_goals_for": home_goals_for,
        "away_recent_goals_for": away_goals_for,
        "home_recent_goals_against": home_goals_against,
        "away_recent_goals_against": away_goals_against,
        "home_goals_for_last_5": float(home.get("goals_for_last_5", 0.0)),
        "away_goals_for_last_5": float(away.get("goals_for_last_5", 0.0)),
        "home_goals_against_last_5": float(home.get("goals_against_last_5", 0.0)),
        "away_goals_against_last_5": float(away.get("goals_against_last_5", 0.0)),
        "home_goals_for_last_10": float(home.get("goals_for_last_10", 0.0)),
        "away_goals_for_last_10": float(away.get("goals_for_last_10", 0.0)),
        "home_goals_against_last_10": float(home.get("goals_against_last_10", 0.0)),
        "away_goals_against_last_10": float(away.get("goals_against_last_10", 0.0)),
        "expected_goal_total_proxy": (
            home_goals_for + away_goals_for + home_goals_against + away_goals_against
        )
        / 2,
        "home_low_scoring_rate": home_low_scoring,
        "away_low_scoring_rate": away_low_scoring,
        "low_scoring_average": (home_low_scoring + away_low_scoring) / 2,
        "home_clean_sheet_rate": float(home.get("clean_sheet_rate", 0.0)),
        "away_clean_sheet_rate": float(away.get("clean_sheet_rate", 0.0)),
        "neutral": int(neutral),
        **_competition_features(tournament),
        "squad_avg_age_diff": diff("squad_avg_age"),
        "squad_avg_caps_diff": diff("squad_avg_caps"),
        "squad_total_caps_diff": diff("squad_total_caps"),
        "squad_club_strength_diff": diff("squad_club_strength"),
        "squad_attack_diff": diff("squad_attack"),
        "squad_defense_diff": diff("squad_defense"),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
