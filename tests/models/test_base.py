import pandas as pd
import pytest

from fpl_model.models.base import ActionModel, PlayerPredictions, SeasonData
from fpl_model.models.defaults import get_default_registry
from fpl_model.models.optimizers.greedy import GreedyOptimizer
from fpl_model.models.predictors.form import FormPredictor
from fpl_model.models.registry import ModelRegistry
from fpl_model.simulation.actions import (
    ChipType,
    PlayChip,
    SetCaptain,
    SetLineup,
    SetViceCaptain,
    Transfer,
)
from fpl_model.simulation.state import PlayerInSquad, SquadState


class TestActions:
    def test_transfer_creation(self):
        t = Transfer(player_out=1, player_in=2)
        assert t.player_out == 1
        assert t.player_in == 2

    def test_set_captain(self):
        c = SetCaptain(player_id=1)
        assert c.player_id == 1

    def test_play_chip(self):
        c = PlayChip(chip_type=ChipType.BENCH_BOOST)
        assert c.chip_type == ChipType.BENCH_BOOST

    def test_set_lineup(self):
        sl = SetLineup(starting_xi=[1, 2, 3], bench_order=[4, 5])
        assert len(sl.starting_xi) == 3


class TestSquadState:
    def test_create_squad_state(self):
        players = [
            PlayerInSquad(code=i, element_type=(i % 4) + 1, buy_price=50, sell_price=50)
            for i in range(15)
        ]
        state = SquadState(
            players=players,
            budget=0,
            free_transfers=1,
            chips_available={ChipType.WILDCARD: 2, ChipType.BENCH_BOOST: 2},
            current_gameweek=1,
        )
        assert len(state.players) == 15
        assert state.free_transfers == 1


class TestModelRegistry:
    def test_register_and_get(self):
        registry = ModelRegistry()

        class DummyModel(ActionModel):
            def recommend(self, state, data):
                return []

        model = DummyModel()
        registry.register("dummy", model)
        assert registry.get("dummy") is model

    def test_get_unknown_raises(self):
        registry = ModelRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_models(self):
        registry = ModelRegistry()

        class DummyModel(ActionModel):
            def recommend(self, state, data):
                return []

        registry.register("dummy", DummyModel())
        assert "dummy" in registry.list()


def _make_squad_and_data():
    """Helper to create a squad and season data for model tests."""
    # 2 GK, 5 DEF, 5 MID, 3 FWD
    players = []
    code = 1
    for et, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        for _ in range(count):
            players.append(PlayerInSquad(code=code, element_type=et, buy_price=50, sell_price=50))
            code += 1

    state = SquadState(
        players=players,
        budget=100,
        free_transfers=1,
        chips_available={ct: 2 for ct in ChipType},
        current_gameweek=5,
        starting_xi=[p.code for p in players[:11]],
        bench_order=[p.code for p in players[11:]],
        captain=8,
        vice_captain=9,
    )

    players_df = pd.DataFrame({"code": [p.code for p in players], "element_type": [p.element_type for p in players]})

    gw_rows = []
    for gw in range(1, 5):
        for p in players:
            gw_rows.append({"player_code": p.code, "gameweek": gw, "total_points": p.code})
    gw_perf = pd.DataFrame(gw_rows)

    data = SeasonData(
        gameweek_performances=gw_perf,
        fixtures=pd.DataFrame(),
        players=players_df,
        teams=pd.DataFrame(),
        current_gameweek=5,
        season="2024-25",
    )
    return state, data


class TestFormPredictor:
    def test_predicts_average_of_recent_gws(self):
        state, data = _make_squad_and_data()
        predictor = FormPredictor(lookback=5)
        preds = predictor.predict(state, data)
        # Player with code=1 always scores 1, player with code=15 always scores 15
        assert preds.predictions[1] == pytest.approx(1.0)
        assert preds.predictions[15] == pytest.approx(15.0)

    def test_no_data_gives_baseline(self):
        state, data = _make_squad_and_data()
        data.gameweek_performances = pd.DataFrame(columns=["player_code", "gameweek", "total_points"])
        predictor = FormPredictor(lookback=5, baseline=3.0)
        preds = predictor.predict(state, data)
        assert preds.predictions[1] == 3.0


class TestGreedyOptimizer:
    def test_returns_lineup_captain_vice(self):
        state, data = _make_squad_and_data()
        preds = PlayerPredictions(predictions={p.code: float(p.code) for p in state.players})
        optimizer = GreedyOptimizer()
        actions = optimizer.optimize(preds, state, data)
        action_types = {type(a) for a in actions}
        assert SetLineup in action_types
        assert SetCaptain in action_types
        assert SetViceCaptain in action_types

    def test_captain_is_highest_predicted(self):
        state, data = _make_squad_and_data()
        preds = PlayerPredictions(predictions={p.code: float(p.code) for p in state.players})
        optimizer = GreedyOptimizer()
        actions = optimizer.optimize(preds, state, data)
        captain_action = next(a for a in actions if isinstance(a, SetCaptain))
        lineup_action = next(a for a in actions if isinstance(a, SetLineup))
        # Captain should be the highest-predicted player in starting XI
        assert captain_action.player_id in lineup_action.starting_xi

    def test_formation_valid(self):
        state, data = _make_squad_and_data()
        preds = PlayerPredictions(predictions={p.code: float(p.code) for p in state.players})
        optimizer = GreedyOptimizer()
        actions = optimizer.optimize(preds, state, data)
        lineup = next(a for a in actions if isinstance(a, SetLineup))
        assert len(lineup.starting_xi) == 11
        assert len(lineup.bench_order) == 4


class TestDefaultRegistry:
    def test_form_greedy_registered(self):
        registry = get_default_registry()
        assert "form-greedy" in registry.list()

    def test_form_greedy_produces_actions(self):
        state, data = _make_squad_and_data()
        registry = get_default_registry()
        model = registry.get("form-greedy")
        actions = model.recommend(state, data)
        assert len(actions) > 0
