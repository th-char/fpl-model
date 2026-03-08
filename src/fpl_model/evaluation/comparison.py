"""Compare simulation results across multiple models."""

from __future__ import annotations

from fpl_model.evaluation.metrics import compute_metrics
from fpl_model.simulation.engine import SimulationResult


def compare_results(results: dict[str, SimulationResult]) -> list[dict]:
    """Compare multiple simulation results, returning rows sorted by total points."""
    rows = []
    for name, result in results.items():
        metrics = compute_metrics(result)
        rows.append({"name": name, **metrics})
    rows.sort(key=lambda r: r["total_points"], reverse=True)
    return rows
