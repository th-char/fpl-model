"""Registry for named model instances."""

from __future__ import annotations

from fpl_model.models.base import ActionModel


class ModelRegistry:
    """Simple dict-based registry for ActionModel instances."""

    def __init__(self) -> None:
        self._models: dict[str, ActionModel] = {}

    def register(self, name: str, model: ActionModel) -> None:
        """Register a model under the given name."""
        self._models[name] = model

    def get(self, name: str) -> ActionModel:
        """Retrieve a model by name. Raises KeyError if not found."""
        return self._models[name]

    def list(self) -> list[str]:
        """Return all registered model names."""
        return list(self._models.keys())
