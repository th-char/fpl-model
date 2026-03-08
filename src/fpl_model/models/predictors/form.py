"""Form-based predictor: predicts points using recent gameweek averages."""

from __future__ import annotations

from fpl_model.models.base import PlayerPredictions, Predictor, SeasonData
from fpl_model.simulation.state import SquadState

DEFAULT_LOOKBACK = 5
DEFAULT_BASELINE = 2.0


class FormPredictor(Predictor):
    """Predict player points as the average of their last N gameweeks."""

    def __init__(self, lookback: int = DEFAULT_LOOKBACK, baseline: float = DEFAULT_BASELINE) -> None:
        self.lookback = lookback
        self.baseline = baseline

    def predict(self, state: SquadState, data: SeasonData) -> PlayerPredictions:
        """Return point predictions based on recent form.

        For each player in ``data.players``, compute the mean ``total_points``
        over the most recent *lookback* gameweeks from
        ``data.gameweek_performances``.  Players with no recent data receive
        the *baseline* score.
        """
        gw_perf = data.gameweek_performances
        current_gw = data.current_gameweek

        # Handle empty DataFrame (e.g., initial state before any GWs)
        if gw_perf.empty or "gameweek" not in gw_perf.columns:
            predictions = {int(c): self.baseline for c in data.players["code"]}
            return PlayerPredictions(predictions=predictions)

        # Filter to recent gameweeks (before current)
        min_gw = max(1, current_gw - self.lookback)
        recent = gw_perf[(gw_perf["gameweek"] >= min_gw) & (gw_perf["gameweek"] < current_gw)]

        # Compute mean total_points per player_code
        if len(recent) > 0:
            avg_points = recent.groupby("player_code")["total_points"].mean()
            avg_map: dict[int, float] = avg_points.to_dict()
        else:
            avg_map = {}

        # Build predictions for all players in the players DataFrame
        predictions: dict[int, float] = {}
        for code in data.players["code"]:
            predictions[int(code)] = avg_map.get(int(code), self.baseline)

        return PlayerPredictions(predictions=predictions)
