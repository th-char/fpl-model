"""Evaluation metrics for simulation results."""

from __future__ import annotations

from fpl_model.simulation.engine import SimulationResult


def compute_metrics(result: SimulationResult) -> dict:
    """Compute summary metrics from a simulation result."""
    gw_points = result.gameweek_points
    num_gws = len(gw_points)
    best_gw = max(gw_points.items(), key=lambda x: x[1])
    worst_gw = min(gw_points.items(), key=lambda x: x[1])
    return {
        "total_points": result.total_points,
        "num_gameweeks": num_gws,
        "avg_points_per_gw": result.total_points / num_gws if num_gws > 0 else 0,
        "total_transfer_cost": sum(result.transfer_costs.values()),
        "best_gw": best_gw,
        "worst_gw": worst_gw,
    }
