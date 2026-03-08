import pytest

from fpl_model.evaluation.comparison import compare_results
from fpl_model.evaluation.metrics import compute_metrics
from fpl_model.evaluation.reports import format_report
from fpl_model.simulation.engine import SimulationResult


class TestMetrics:
    def test_compute_metrics(self):
        result = SimulationResult(
            total_points=200,
            gameweek_points={1: 60, 2: 70, 3: 70},
            actions_log={},
            transfer_costs={1: 0, 2: 4, 3: 0},
            budget_history={1: 100, 2: 95, 3: 90},
        )
        metrics = compute_metrics(result)
        assert metrics["total_points"] == 200
        assert metrics["num_gameweeks"] == 3
        assert metrics["avg_points_per_gw"] == pytest.approx(200 / 3, rel=0.01)
        assert metrics["total_transfer_cost"] == 4
        assert metrics["best_gw"] == (2, 70)  # first GW with max
        assert metrics["worst_gw"] == (1, 60)


class TestComparison:
    def test_compare_two_models(self):
        r1 = SimulationResult(
            total_points=200,
            gameweek_points={1: 60, 2: 70, 3: 70},
            actions_log={},
            transfer_costs={1: 0, 2: 0, 3: 0},
            budget_history={},
        )
        r2 = SimulationResult(
            total_points=180,
            gameweek_points={1: 50, 2: 60, 3: 70},
            actions_log={},
            transfer_costs={1: 0, 2: 4, 3: 0},
            budget_history={},
        )
        comparison = compare_results({"model_a": r1, "model_b": r2})
        assert comparison[0]["name"] == "model_a"
        assert comparison[0]["total_points"] == 200


class TestReports:
    def test_format_report_returns_string(self):
        metrics = {
            "total_points": 200,
            "num_gameweeks": 3,
            "avg_points_per_gw": 66.7,
            "total_transfer_cost": 4,
            "best_gw": (2, 70),
            "worst_gw": (1, 60),
        }
        report = format_report("test_model", metrics)
        assert "test_model" in report
        assert "200" in report
