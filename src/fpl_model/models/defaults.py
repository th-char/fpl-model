"""Default model registry with built-in models."""

from __future__ import annotations

from fpl_model.data.db import Database
from fpl_model.models.registry import ModelRegistry


def get_default_registry() -> ModelRegistry:
    """Create a registry pre-loaded with built-in models.

    Note: ``ppo-agent`` is not included here because it requires a Database
    and season list at construction time. Use ``create_ppo_agent()`` instead.
    """
    from fpl_model.models.base import PredictOptimizeModel
    from fpl_model.models.optimizers.greedy import GreedyOptimizer
    from fpl_model.models.optimizers.lp_optimizer import LPOptimizer
    from fpl_model.models.predictors.form import FormPredictor
    from fpl_model.models.predictors.sequence_predictor import SequencePredictor
    from fpl_model.models.predictors.xgboost_predictor import XGBoostPredictor

    registry = ModelRegistry()

    # Baseline: simple form average + lineup-only optimizer
    registry.register("form-greedy", PredictOptimizeModel(FormPredictor(), GreedyOptimizer()))

    # XGBoost + greedy with transfers
    registry.register(
        "xgb-greedy",
        PredictOptimizeModel(XGBoostPredictor(), GreedyOptimizer(enable_transfers=True)),
    )

    # XGBoost + LP optimizer
    registry.register("xgb-lp", PredictOptimizeModel(XGBoostPredictor(), LPOptimizer()))

    # LSTM sequence + LP optimizer
    registry.register("sequence-lp", PredictOptimizeModel(SequencePredictor(), LPOptimizer()))

    return registry


def create_ppo_agent(
    db: Database, seasons: list[str], **kwargs
) -> "PPOAgent":  # noqa: F821
    """Factory for PPOAgent which requires runtime dependencies (db + seasons).

    Parameters
    ----------
    db : Database
        Database instance with historical data.
    seasons : list[str]
        Season identifiers to train on.
    **kwargs
        Additional keyword arguments passed to ``PPOAgent.__init__``.
    """
    from fpl_model.models.rl.ppo import PPOAgent

    return PPOAgent(db=db, seasons=seasons, **kwargs)
