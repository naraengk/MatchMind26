from pathlib import Path

from wc26_predictor.data.loading import read_matches, read_qualified_teams, read_squads
from wc26_predictor.features.build_features import (
    FEATURE_COLUMNS,
    HISTORICAL_FEATURE_COLUMNS,
    build_prediction_row,
    build_team_profiles,
    build_training_frame,
)


ROOT = Path(__file__).resolve().parents[1]


def test_build_training_frame_contains_expected_features() -> None:
    matches = read_matches(ROOT / "data/sample/matches.csv")
    squads = read_squads(ROOT / "data/sample/squads.csv")

    frame = build_training_frame(matches, squads)

    assert not frame.empty
    assert set(FEATURE_COLUMNS).issubset(frame.columns)
    assert set(frame["target"]).issubset({"home_win", "draw", "away_win"})
    assert {"home_goals", "away_goals"}.issubset(frame.columns)
    assert "abs_elo_diff" in HISTORICAL_FEATURE_COLUMNS
    assert "draw_rate_average" in HISTORICAL_FEATURE_COLUMNS
    assert not any(column.startswith("squad_") for column in HISTORICAL_FEATURE_COLUMNS)


def test_build_prediction_row_for_known_teams() -> None:
    matches = read_matches(ROOT / "data/sample/matches.csv")
    squads = read_squads(ROOT / "data/sample/squads.csv")
    profiles = build_team_profiles(matches, squads)

    row = build_prediction_row("United States", "England", profiles)

    assert list(row.columns) == FEATURE_COLUMNS
    assert row.shape == (1, len(FEATURE_COLUMNS))
    assert row["abs_elo_diff"].item() == abs(row["elo_diff"].item())



def test_squad_ages_are_calculated_for_2026_opener() -> None:
    squads = read_squads(ROOT / "data/sample/squads.csv")
    messi_age = squads.loc[squads["player"] == "Lionel Messi", "age"].item()
    bellingham_age = squads.loc[squads["player"] == "Jude Bellingham", "age"].item()

    assert messi_age == 38
    assert bellingham_age == 22


def test_qualified_team_profiles_include_all_48_teams() -> None:
    matches = read_matches(ROOT / "data/sample/matches.csv")
    squads = read_squads(ROOT / "data/sample/squads.csv")
    qualified = read_qualified_teams(ROOT / "data/sample/qualified_teams.csv")

    profiles = build_team_profiles(matches, squads, team_pool=qualified["team"].tolist())

    assert len(profiles) == 48
    assert "Iraq" in profiles
    assert "DR Congo" in profiles
    assert "Italy" not in profiles
