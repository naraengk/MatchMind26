from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBClassifier


CLASSES = np.array(["away_win", "draw", "home_win"])


class MatchOutcomeEnsemble:
    """Two-stage outcome classifier blended with independent Poisson goal models."""

    def __init__(
        self,
        feature_columns: list[str],
        draw_threshold: float = 0.30,
        outcome_weight: float = 0.65,
        random_state: int = 42,
    ) -> None:
        self.feature_columns = feature_columns
        # If a draw's probability is at least this high, we call the match a draw.
        self.draw_threshold = draw_threshold
        self.outcome_weight = outcome_weight
        self.random_state = random_state
        self.classes_ = CLASSES.copy()
        self.apply_current_squad_adjustment = False

        # Main model: predicts win / draw / loss directly. Kept shallow and well
        # regularized on purpose so it generalizes instead of memorizing the past.
        self.outcome_model = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            n_estimators=700,
            max_depth=1,
            min_child_weight=6,
            learning_rate=0.035,
            subsample=0.90,
            colsample_bytree=0.85,
            reg_lambda=3,
            reg_alpha=0.0,
            random_state=random_state,
            n_jobs=8,
        )
        # Two goal models (one per side) used for the projected score. "Poisson"
        # loss fits goal counts, which are whole numbers that cluster around a few.
        self.home_goals_model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.045,
            max_iter=250,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            random_state=random_state + 2,
        )
        self.away_goals_model = HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.045,
            max_iter=250,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            random_state=random_state + 3,
        )

    def fit(self, x: pd.DataFrame, y: pd.Series, home_goals: pd.Series, away_goals: pd.Series):
        # Train all three models: one for the result, two for the goals.
        features = x[self.feature_columns]
        encoded = y.map({"away_win": 0, "draw": 1, "home_win": 2})
        self.outcome_model.fit(features, encoded)
        self.home_goals_model.fit(features, home_goals)
        self.away_goals_model.fit(features, away_goals)
        return self

    def _outcome_probabilities(self, x: pd.DataFrame) -> np.ndarray:
        # The classifier's direct win/draw/loss probabilities.
        features = x[self.feature_columns]
        return self.outcome_model.predict_proba(features)

    def _score_probabilities(self, x: pd.DataFrame) -> np.ndarray:
        # Turn the two expected-goal numbers into win/draw/loss odds.
        features = x[self.feature_columns]
        # Predicted average goals for each side, clipped to 0.15..5 goals.
        expected_home = np.clip(self.home_goals_model.predict(features), 0.15, 5.0)
        expected_away = np.clip(self.away_goals_model.predict(features), 0.15, 5.0)
        score_probabilities = []
        score_range = np.arange(0, 9)  # consider scorelines from 0 up to 8 goals
        for home_mean, away_mean in zip(expected_home, expected_away):
            # Chance of each exact score for each team (Poisson distribution)...
            home_distribution = poisson.pmf(score_range, home_mean)
            away_distribution = poisson.pmf(score_range, away_mean)
            # ...combined into a grid of every home-vs-away scoreline.
            matrix = np.outer(home_distribution, away_distribution)
            # Add up the grid: away wins above the diagonal, draws on it, home below.
            away_win = float(np.triu(matrix, 1).sum())
            draw = float(np.trace(matrix))
            home_win = float(np.tril(matrix, -1).sum())
            total = max(away_win + draw + home_win, 1e-9)
            score_probabilities.append([away_win / total, draw / total, home_win / total])
        return np.asarray(score_probabilities)

    @staticmethod
    def _squad_edge(row: pd.Series) -> float:
        # Roll the 2026 squad-quality gaps into one small number between about -0.08 and 0.08.
        values = [
            float(row.get("squad_club_strength_diff", 0.0)) / 18,
            float(row.get("squad_attack_diff", 0.0)) / 140,
            float(row.get("squad_defense_diff", 0.0)) / 140,
            float(row.get("squad_total_caps_diff", 0.0)) / 350,
        ]
        return math.tanh(float(np.mean(values))) * 0.08

    def _apply_squad_adjustment(self, probabilities: np.ndarray, x: pd.DataFrame) -> np.ndarray:
        # Only used for live 2026 predictions, never during historical testing.
        if not self.apply_current_squad_adjustment:
            return probabilities
        adjusted = probabilities.copy()
        for index, (_, row) in enumerate(x.iterrows()):
            # Nudge the better squad's win chance up and the other's down, then renormalize.
            edge = self._squad_edge(row)
            adjusted[index, 2] *= 1 + max(edge, -0.06)
            adjusted[index, 0] *= 1 - min(edge, 0.06)
            adjusted[index, 1] *= 1 - min(abs(edge) * 0.35, 0.025)
            adjusted[index] /= adjusted[index].sum()
        return adjusted

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        # Final win/draw/loss probabilities for each match.
        probabilities = self._outcome_probabilities(x)
        return self._apply_squad_adjustment(probabilities, x)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        # Pick a single result from the probabilities.
        probabilities = self.predict_proba(x)
        draw_index = list(self.classes_).index("draw")
        predictions = []
        for row in probabilities:
            # Call a draw only if its probability clears the tuned threshold,
            # otherwise take whichever of the two teams is more likely to win.
            if row[draw_index] >= self.draw_threshold:
                predictions.append("draw")
                continue
            decisive = row.copy()
            decisive[draw_index] = -1
            predictions.append(str(self.classes_[int(np.argmax(decisive))]))
        return np.asarray(predictions)

    def expected_goals(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # The projected scoreline shown in the app.
        features = x[self.feature_columns]
        return (
            np.clip(self.home_goals_model.predict(features), 0.15, 5.0),
            np.clip(self.away_goals_model.predict(features), 0.15, 5.0),
        )
