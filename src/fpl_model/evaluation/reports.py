"""Report formatting for evaluation results."""

from __future__ import annotations


def format_report(model_name: str, metrics: dict) -> str:
    """Format a single model's metrics as a human-readable report."""
    lines = [
        f"=== {model_name} ===",
        f"Total Points: {metrics['total_points']}",
        f"Gameweeks: {metrics['num_gameweeks']}",
        f"Avg Points/GW: {metrics['avg_points_per_gw']:.1f}",
        f"Transfer Costs: {metrics['total_transfer_cost']}",
        f"Best GW: GW{metrics['best_gw'][0]} ({metrics['best_gw'][1]} pts)",
        f"Worst GW: GW{metrics['worst_gw'][0]} ({metrics['worst_gw'][1]} pts)",
    ]
    return "\n".join(lines)


def format_comparison(comparison: list[dict]) -> str:
    """Format a comparison table as a human-readable report."""
    lines = ["=== Model Comparison ===", ""]
    header = f"{'Rank':<6}{'Model':<20}{'Points':<10}{'Avg/GW':<10}{'Hits':<8}"
    lines.append(header)
    lines.append("-" * len(header))
    for i, row in enumerate(comparison, 1):
        lines.append(
            f"{i:<6}{row['name']:<20}{row['total_points']:<10}"
            f"{row['avg_points_per_gw']:<10.1f}{row['total_transfer_cost']:<8}"
        )
    return "\n".join(lines)
