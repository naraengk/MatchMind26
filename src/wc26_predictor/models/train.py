from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    log_loss,
    mean_absolute_error,
    recall_score,
)

from wc26_predictor.data.loading import read_matches, read_qualified_teams, read_squads
from wc26_predictor.features.build_features import (
    FEATURE_COLUMNS,
    HISTORICAL_FEATURE_COLUMNS,
    build_team_profiles,
    build_training_frame,
)
from wc26_predictor.features.rankings import load_ranking_lookup
from wc26_predictor.models.ensemble import MatchOutcomeEnsemble

DEFAULT_RANKINGS_PATH = "data/sample/fifa_rankings.csv"

# Day before the 2026 World Cup opener (June 11). Everything up to and including
# this date — the last friendlies and qualifiers before the tournament — is used
# for training and testing; the tournament itself is never used.
MODEL_DATA_CUTOFF = pd.Timestamp("2026-06-10")


def chronological_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Split by time, not at random: learn on older games, tune on 2024, and test on
    # 2025+ which the model never sees during training. That mimics real forecasting.
    train = frame[frame["date"] < pd.Timestamp("2024-01-01")].copy()
    validation = frame[
        (frame["date"] >= pd.Timestamp("2024-01-01"))
        & (frame["date"] < pd.Timestamp("2025-01-01"))
    ].copy()
    test = frame[frame["date"] >= pd.Timestamp("2025-01-01")].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological train, validation, and test periods must all contain matches.")
    return train, validation, test


def _draw_recall(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(
        recall_score(y_true, y_pred, labels=["draw"], average=None, zero_division=0)[0]
    )


def _tune_ensemble(
    model: MatchOutcomeEnsemble,
    validation: pd.DataFrame,
) -> dict[str, float]:
    # Find the draw cut-off that scores best on 2024 (draws are the hardest call).
    best: dict[str, float] | None = None
    probabilities = model.predict_proba(validation)
    for threshold in np.arange(0.20, 0.43, 0.01):
        model.draw_threshold = float(threshold)
        predictions = model.predict(validation)
        accuracy = float(accuracy_score(validation["target"], predictions))
        draw_recall = _draw_recall(validation["target"], predictions)
        macro_f1 = float(f1_score(validation["target"], predictions, average="macro"))
        if draw_recall < 0.30:
            continue
        overprediction_penalty = max(draw_recall - 0.48, 0) * 0.4
        objective = 0.75 * accuracy + 0.25 * macro_f1 - overprediction_penalty
        candidate = {
            "objective": objective,
            "draw_threshold": float(threshold),
            "accuracy": accuracy,
            "draw_recall": draw_recall,
            "macro_f1": macro_f1,
            "log_loss": float(
                log_loss(validation["target"], probabilities, labels=model.classes_)
            ),
        }
        if best is None or candidate["objective"] > best["objective"]:
            best = candidate
    if best is None:
        raise RuntimeError("Could not tune ensemble.")
    model.draw_threshold = best["draw_threshold"]
    return best


def _calibration_error(
    y_true: pd.Series, probabilities: np.ndarray, classes: np.ndarray, bins: int = 10
) -> float:
    predicted = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(len(probabilities)), predicted]
    correct = classes[predicted] == y_true.to_numpy()
    error = 0.0
    for lower in np.linspace(0, 1, bins, endpoint=False):
        upper = lower + 1 / bins
        mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            error += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return error


def evaluate_model(model: MatchOutcomeEnsemble, frame: pd.DataFrame) -> dict:
    probabilities = model.predict_proba(frame)
    predictions = model.predict(frame)
    y = frame["target"]
    one_hot = np.column_stack([y.eq(label).astype(int) for label in model.classes_])
    home_goals, away_goals = model.expected_goals(frame)
    report_dict = classification_report(y, predictions, zero_division=0, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "draw_recall": _draw_recall(y, predictions),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "log_loss": float(log_loss(y, probabilities, labels=model.classes_)),
        "multiclass_brier_score": float(
            np.mean(
                [
                    brier_score_loss(one_hot[:, index], probabilities[:, index])
                    for index in range(len(model.classes_))
                ]
            )
        ),
        "expected_calibration_error": _calibration_error(
            y, probabilities, model.classes_
        ),
        "home_goals_mae": float(mean_absolute_error(frame["home_goals"], home_goals)),
        "away_goals_mae": float(mean_absolute_error(frame["away_goals"], away_goals)),
        "classification_report": classification_report(y, predictions, zero_division=0),
        "classification_report_dict": report_dict,
        "sample_count": int(len(frame)),
        "class_counts": {key: int(value) for key, value in y.value_counts().items()},
    }


def train_model(
    matches_path: Union[str, Path],
    squads_path: Union[str, Path],
    model_out: Union[str, Path],
    qualified_teams_path: Union[str, Path, None] = None,
    rankings_path: Union[str, Path, None] = DEFAULT_RANKINGS_PATH,
) -> dict:
    """Build the features, train and score the model, and save it to disk."""
    # Load everything and drop any matches after the cutoff (the tournament itself).
    matches = read_matches(matches_path)
    matches = matches[matches["date"] <= MODEL_DATA_CUTOFF].copy()
    squads = read_squads(squads_path)
    rankings = (
        load_ranking_lookup(rankings_path)
        if rankings_path is not None and Path(rankings_path).exists()
        else None
    )
    qualified_teams = (
        read_qualified_teams(qualified_teams_path) if qualified_teams_path is not None else None
    )
    team_pool = qualified_teams["team"].tolist() if qualified_teams is not None else None
    frame = build_training_frame(matches, squads, rankings)
    train, validation, test = chronological_split(frame)

    model = MatchOutcomeEnsemble(feature_columns=HISTORICAL_FEATURE_COLUMNS)
    model.fit(
        train,
        train["target"],
        train["home_goals"],
        train["away_goals"],
    )
    tuning = _tune_ensemble(model, validation)
    validation_metrics = evaluate_model(model, validation)
    test_metrics = evaluate_model(model, test)

    majority_accuracy = float(test["target"].value_counts(normalize=True).max())
    metrics = {
        **test_metrics,
        "baseline_accuracy": majority_accuracy,
        "accuracy_lift_over_baseline": test_metrics["accuracy"] - majority_accuracy,
        "validation": validation_metrics,
        "tuning": tuning,
        "split": {
            "train": "Before 2024-01-01",
            "validation": "2024-01-01 through 2024-12-31",
            "test": "2025-01-01 onward",
            "train_samples": int(len(train)),
            "validation_samples": int(len(validation)),
            "test_samples": int(len(test)),
        },
        "feature_columns": FEATURE_COLUMNS,
        "historical_feature_columns": HISTORICAL_FEATURE_COLUMNS,
        "squad_adjustment": (
            "Current squad quality is excluded from historical evaluation and applied only as "
            "a capped post-model adjustment for 2026 predictions."
        ),
        "model_type": "XGBoost outcome classifier with Poisson expected-goals models",
        "data_cutoff": MODEL_DATA_CUTOFF.strftime("%Y-%m-%d"),
        "ranking_note": (
            "FIFA ranking difference is included as a leakage-safe feature: each match uses the "
            "most recent ranking published before it was played. Combined with Elo it improves "
            "draw recall and log loss on the held-out test set."
        ),
    }

    # The model we ship is retrained on train + 2024 (more data) using the draw
    # threshold we just tuned, and is allowed to use the current-squad adjustment.
    deployment_frame = pd.concat([train, validation], ignore_index=True)
    deployment_model = MatchOutcomeEnsemble(
        feature_columns=HISTORICAL_FEATURE_COLUMNS,
        draw_threshold=model.draw_threshold,
    )
    deployment_model.fit(
        deployment_frame,
        deployment_frame["target"],
        deployment_frame["home_goals"],
        deployment_frame["away_goals"],
    )
    deployment_model.apply_current_squad_adjustment = True
    artifact = {
        "model": deployment_model,
        "classes": list(deployment_model.classes_),
        "team_profiles": build_team_profiles(
            matches, squads, team_pool=team_pool, rankings=rankings
        ),
        "qualified_teams": team_pool,
        "metrics": metrics,
    }
    model_out = Path(model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_out)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the WC 2026 match predictor.")
    parser.add_argument("--matches", required=True, help="Path to historical matches CSV.")
    parser.add_argument("--squads", required=True, help="Path to squad/player features CSV.")
    parser.add_argument("--model-out", required=True, help="Where to save the trained model artifact.")
    parser.add_argument(
        "--qualified-teams",
        default=None,
        help="Optional CSV of the 48 qualified teams to expose for prediction.",
    )
    parser.add_argument(
        "--rankings",
        default=DEFAULT_RANKINGS_PATH,
        help="CSV of dated FIFA rankings (team, date, total_points).",
    )
    args = parser.parse_args()

    metrics = train_model(
        args.matches, args.squads, args.model_out, args.qualified_teams, args.rankings
    )
    print(f"accuracy={metrics['accuracy']:.3f}")
    print(f"draw_recall={metrics['draw_recall']:.3f}")
    print(f"macro_f1={metrics['macro_f1']:.3f}")
    print(f"log_loss={metrics['log_loss']:.3f}")
    print(f"baseline_accuracy={metrics['baseline_accuracy']:.3f}")
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()
