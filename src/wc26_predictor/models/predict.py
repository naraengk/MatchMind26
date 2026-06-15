from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import joblib

from wc26_predictor.features.build_features import build_prediction_row


def predict_match(
    model_path: Union[str, Path],
    home_team: str,
    away_team: str,
    neutral: bool = True,
    tournament: str = "FIFA World Cup",
) -> dict[str, Union[float, str]]:
    # Load the saved model + team snapshots and predict one match from the command line.
    artifact = joblib.load(model_path)
    model = artifact["model"]
    row = build_prediction_row(
        home_team=home_team,
        away_team=away_team,
        team_profiles=artifact["team_profiles"],
        neutral=neutral,
        tournament=tournament,
    )
    probabilities = model.predict_proba(row)[0]
    class_probs = dict(zip(model.classes_, probabilities))
    predicted_outcome = str(model.predict(row)[0])
    expected_home, expected_away = model.expected_goals(row)
    return {
        "home_team": home_team,
        "away_team": away_team,
        "predicted_outcome": predicted_outcome,
        "expected_home_goals": round(float(expected_home[0]), 2),
        "expected_away_goals": round(float(expected_away[0]), 2),
        **{label: round(float(prob), 4) for label, prob in class_probs.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a single match outcome.")
    parser.add_argument("--model", required=True, help="Path to trained model artifact.")
    parser.add_argument("--home", required=True, help="Home or listed-first team.")
    parser.add_argument("--away", required=True, help="Away or listed-second team.")
    parser.add_argument("--not-neutral", action="store_true", help="Mark match as not neutral.")
    args = parser.parse_args()

    result = predict_match(args.model, args.home, args.away, neutral=not args.not_neutral)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
