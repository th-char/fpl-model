"""Abstract base classes for FPL prediction and optimization models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd

from fpl_model.simulation.actions import Action
from fpl_model.simulation.state import SquadState


@dataclass
class SeasonData:
    """All data for a single FPL season."""

    gameweek_performances: pd.DataFrame
    fixtures: pd.DataFrame
    players: pd.DataFrame
    teams: pd.DataFrame
    current_gameweek: int
    season: str


@dataclass
class PlayerPredictions:
    """Point predictions for players."""

    predictions: dict[int, float]  # player code -> predicted points
    multi_gw: dict[int, list[float]] | None = None  # optional multi-GW forecasts


@dataclass
class HistoricalData:
    """Collection of season data for training."""

    seasons: dict[str, SeasonData] = field(default_factory=dict)


class ActionModel(ABC):
    """Base class for any model that recommends FPL actions."""

    @abstractmethod
    def recommend(self, state: SquadState, data: SeasonData) -> list[Action]:
        """Given current squad state and season data, return recommended actions."""
        ...

    def train(self, historical_data: HistoricalData) -> None:
        """Optional training step using historical data. Default is no-op."""
        pass


class Predictor(ABC):
    """Predicts expected points for players."""

    @abstractmethod
    def predict(self, state: SquadState, data: SeasonData) -> PlayerPredictions:
        """Return point predictions for available players."""
        ...

    def train(self, historical_data: HistoricalData) -> None:
        """Optional training step. Default is no-op."""
        pass


class Optimizer(ABC):
    """Selects optimal actions given predictions and constraints."""

    @abstractmethod
    def optimize(
        self,
        predictions: PlayerPredictions,
        state: SquadState,
        data: SeasonData,
    ) -> list[Action]:
        """Return optimal actions given predictions and squad state."""
        ...


class PredictOptimizeModel(ActionModel):
    """Composes a Predictor and Optimizer into a full ActionModel."""

    def __init__(self, predictor: Predictor, optimizer: Optimizer) -> None:
        self.predictor = predictor
        self.optimizer = optimizer

    def recommend(self, state: SquadState, data: SeasonData) -> list[Action]:
        """Predict player points, then optimize actions."""
        predictions = self.predictor.predict(state, data)
        return self.optimizer.optimize(predictions, state, data)

    def train(self, historical_data: HistoricalData) -> None:
        """Train the predictor component."""
        self.predictor.train(historical_data)
