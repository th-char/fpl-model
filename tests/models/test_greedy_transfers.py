import pandas as pd
import pytest

from fpl_model.models.base import PlayerPredictions, SeasonData
from fpl_model.models.optimizers.greedy import GreedyOptimizer
from fpl_model.simulation.actions import ChipType, Transfer
from fpl_model.simulation.state import PlayerInSquad, SquadState


def _make_state_and_data():
    """Create a squad and data where a transfer is clearly beneficial."""
    # Squad: 2 GK, 5 DEF, 5 MID, 3 FWD
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

    # Available players include a much better option (code=99)
    all_players = pd.DataFrame({
        "code": [p.code for p in players] + [99],
        "element_type": [p.element_type for p in players] + [3],  # midfielder
        "team_code": list(range(1, 16)) + [20],
        "now_cost": [50] * 15 + [50],
    })

    data = SeasonData(
        gameweek_performances=pd.DataFrame(),
        fixtures=pd.DataFrame(),
        players=all_players,
        teams=pd.DataFrame(),
        current_gameweek=5,
        season="2024-25",
    )

    return state, data


class TestGreedyTransfers:
    def test_makes_transfer_when_gain_exceeds_threshold(self):
        state, data = _make_state_and_data()
        # Player 99 (available MID) predicted at 10.0, worst squad MID at 1.0
        preds = {p.code: 3.0 for p in state.players}
        preds[12] = 1.0  # a midfielder in squad, predicted low
        preds[99] = 10.0  # available midfielder, predicted high
        predictions = PlayerPredictions(predictions=preds)

        optimizer = GreedyOptimizer(enable_transfers=True, transfer_gain_threshold=2.0)
        actions = optimizer.optimize(predictions, state, data)

        transfers = [a for a in actions if isinstance(a, Transfer)]
        assert len(transfers) == 1
        assert transfers[0].player_out == 12
        assert transfers[0].player_in == 99

    def test_no_transfer_when_gain_below_threshold(self):
        state, data = _make_state_and_data()
        # All players predicted similarly
        preds = {p.code: 5.0 for p in state.players}
        preds[99] = 5.5  # marginal improvement
        predictions = PlayerPredictions(predictions=preds)

        optimizer = GreedyOptimizer(enable_transfers=True, transfer_gain_threshold=2.0)
        actions = optimizer.optimize(predictions, state, data)

        transfers = [a for a in actions if isinstance(a, Transfer)]
        assert len(transfers) == 0

    def test_no_transfer_when_disabled(self):
        state, data = _make_state_and_data()
        preds = {p.code: 3.0 for p in state.players}
        preds[12] = 1.0
        preds[99] = 10.0
        predictions = PlayerPredictions(predictions=preds)

        optimizer = GreedyOptimizer(enable_transfers=False)
        actions = optimizer.optimize(predictions, state, data)
        transfers = [a for a in actions if isinstance(a, Transfer)]
        assert len(transfers) == 0

    def test_respects_budget(self):
        state, data = _make_state_and_data()
        # Make player 99 very expensive
        data.players.loc[data.players["code"] == 99, "now_cost"] = 9999
        preds = {p.code: 3.0 for p in state.players}
        preds[12] = 1.0
        preds[99] = 10.0
        predictions = PlayerPredictions(predictions=preds)

        optimizer = GreedyOptimizer(enable_transfers=True, transfer_gain_threshold=0.0)
        actions = optimizer.optimize(predictions, state, data)
        transfers = [a for a in actions if isinstance(a, Transfer)]
        # Should not transfer in an unaffordable player
        assert all(t.player_in != 99 for t in transfers)

    def test_respects_max_3_per_team(self):
        """Cannot transfer in a player if squad already has 3 from that team."""
        state, data = _make_state_and_data()
        # Set player 99's team_code to 1, and make 3 existing squad players also team 1
        data.players.loc[data.players["code"] == 99, "team_code"] = 1
        data.players.loc[data.players["code"] == 1, "team_code"] = 1
        data.players.loc[data.players["code"] == 3, "team_code"] = 1
        data.players.loc[data.players["code"] == 4, "team_code"] = 1

        preds = {p.code: 3.0 for p in state.players}
        preds[12] = 1.0
        preds[99] = 10.0
        predictions = PlayerPredictions(predictions=preds)

        optimizer = GreedyOptimizer(enable_transfers=True, transfer_gain_threshold=0.0)
        actions = optimizer.optimize(predictions, state, data)
        transfers = [a for a in actions if isinstance(a, Transfer)]
        # Player 99 should NOT be transferred in (team already has 3 players)
        assert all(t.player_in != 99 for t in transfers)

    def test_uses_multiple_free_transfers(self):
        """With 2 free transfers, can make up to 2 beneficial swaps."""
        state, data = _make_state_and_data()
        state.free_transfers = 2

        # Add a second good available player (code=98, forward)
        new_row = pd.DataFrame({
            "code": [98],
            "element_type": [4],
            "team_code": [19],
            "now_cost": [50],
        })
        data.players = pd.concat([data.players, new_row], ignore_index=True)

        preds = {p.code: 3.0 for p in state.players}
        preds[12] = 1.0   # worst MID
        preds[15] = 1.0   # worst FWD
        preds[99] = 10.0  # great MID
        preds[98] = 10.0  # great FWD
        predictions = PlayerPredictions(predictions=preds)

        optimizer = GreedyOptimizer(enable_transfers=True, transfer_gain_threshold=2.0)
        actions = optimizer.optimize(predictions, state, data)

        transfers = [a for a in actions if isinstance(a, Transfer)]
        assert len(transfers) == 2
        out_codes = {t.player_out for t in transfers}
        in_codes = {t.player_in for t in transfers}
        assert out_codes == {12, 15}
        assert in_codes == {98, 99}

    def test_default_init_no_transfers(self):
        """Default GreedyOptimizer() should work exactly as before."""
        state, data = _make_state_and_data()
        preds = {p.code: float(p.code) for p in state.players}
        predictions = PlayerPredictions(predictions=preds)

        optimizer = GreedyOptimizer()
        actions = optimizer.optimize(predictions, state, data)
        transfers = [a for a in actions if isinstance(a, Transfer)]
        assert len(transfers) == 0
