"""Default model registry with built-in models."""

from __future__ import annotations

from fpl_model.models.registry import ModelRegistry


def get_default_registry() -> ModelRegistry:
    """Create a registry pre-loaded with built-in models."""
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
