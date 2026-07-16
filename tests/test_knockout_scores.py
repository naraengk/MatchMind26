from wc26_predictor.api.main import projected_knockout_score


def test_projected_knockout_score_uses_poisson_not_rounding() -> None:
    artifact = {"team_profiles": {"Spain": {"elo": 2100}, "Scotland": {"elo": 1700}}}
    team_one_score, team_two_score = projected_knockout_score(
        "Spain",
        "Scotland",
        "Spain",
        confidence=0.9,
        artifact=artifact,
        expected_home_goals=2.79,
        expected_away_goals=0.64,
    )
    assert (team_one_score, team_two_score) == (2, 0)

    team_one_score, team_two_score = projected_knockout_score(
        "England",
        "Argentina",
        "Argentina",
        confidence=0.6,
        artifact=artifact,
        expected_home_goals=0.97,
        expected_away_goals=1.38,
    )
    assert team_one_score < team_two_score
