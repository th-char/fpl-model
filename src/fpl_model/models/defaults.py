"""Default model registry with built-in baseline models."""

from __future__ import annotations

from fpl_model.models.registry import ModelRegistry


def get_default_registry() -> ModelRegistry:
    """Create a registry pre-loaded with built-in baseline models."""
    from fpl_model.models.base import PredictOptimizeModel
    from fpl_model.models.optimizers.greedy import GreedyOptimizer
    from fpl_model.models.predictors.form import FormPredictor

    registry = ModelRegistry()
    model = PredictOptimizeModel(FormPredictor(), GreedyOptimizer())
    registry.register("form-greedy", model)
    return registry
