import pytest

from fpl_model.models.base import ActionModel
from fpl_model.models.registry import ModelRegistry
from fpl_model.simulation.actions import (
    ChipType,
    PlayChip,
    SetCaptain,
    SetLineup,
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
