from pathlib import Path

from wc26_predictor.models.predict import predict_match
from wc26_predictor.models.train import chronological_split, train_model
from wc26_predictor.data.loading import read_matches, read_squads
from wc26_predictor.features.build_features import build_training_frame


ROOT = Path(__file__).resolve().parents[1]


def test_train_and_predict(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    metrics = train_model(
        ROOT / "data/sample/matches.csv",
        ROOT / "data/sample/squads.csv",
        model_path,
        ROOT / "data/sample/qualified_teams.csv",
    )

    result = predict_match(model_path, "United States", "England")
    new_team_result = predict_match(model_path, "Iraq", "DR Congo")

    assert model_path.exists()
    assert "accuracy" in metrics
    assert "draw_recall" in metrics
    assert "macro_f1" in metrics
    assert "multiclass_brier_score" in metrics
    assert metrics["accuracy"] > metrics["baseline_accuracy"]
    assert result["predicted_outcome"] in {"home_win", "draw", "away_win"}
    assert {"home_win", "draw", "away_win"}.intersection(result)
    assert result["expected_home_goals"] >= 0
    assert result["expected_away_goals"] >= 0
    assert new_team_result["predicted_outcome"] in {"home_win", "draw", "away_win"}


def test_chronological_split_keeps_future_matches_out_of_training() -> None:
    frame = build_training_frame(
        read_matches(ROOT / "data/sample/matches.csv"),
        read_squads(ROOT / "data/sample/squads.csv"),
    )
    train, validation, test = chronological_split(frame)

    assert train["date"].max().year <= 2023
    assert validation["date"].min().year == 2024
    assert validation["date"].max().year == 2024
    assert test["date"].min().year >= 2025
