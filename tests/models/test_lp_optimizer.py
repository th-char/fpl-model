# tests/models/test_lp_optimizer.py
import pandas as pd
import pytest

from fpl_model.models.base import PlayerPredictions, SeasonData
from fpl_model.models.optimizers.lp_optimizer import LPOptimizer
from fpl_model.simulation.actions import (
    ChipType,
    PlayChip,
    SetCaptain,
    SetLineup,
    SetViceCaptain,
    Transfer,
)
from fpl_model.simulation.state import PlayerInSquad, SquadState


def _make_state_and_data():
    """Create a full 15-player squad and available players pool."""
    # Current squad: 2 GK, 5 DEF, 5 MID, 3 FWD
    squad_players = []
    code = 1
    for et, count in [(1, 2), (2, 5), (3, 5), (4, 3)]:
        for _ in range(count):
            squad_players.append(
                PlayerInSquad(code=code, element_type=et, buy_price=50, sell_price=50)
            )
            code += 1

    state = SquadState(
        players=squad_players,
        budget=100,  # 10.0m remaining
        free_transfers=1,
        chips_available={ct: 2 for ct in ChipType},
        current_gameweek=5,
        starting_xi=[p.code for p in squad_players[:11]],
        bench_order=[p.code for p in squad_players[11:]],
        captain=8,
        vice_captain=9,
    )

    # Available pool: squad players + 5 extras (one per position + extras)
    extra_players = [
        {"code": 100, "element_type": 1, "team_code": 16, "now_cost": 45},
        {"code": 101, "element_type": 2, "team_code": 17, "now_cost": 55},
        {"code": 102, "element_type": 3, "team_code": 18, "now_cost": 60},
        {"code": 103, "element_type": 4, "team_code": 19, "now_cost": 70},
        {"code": 104, "element_type": 3, "team_code": 20, "now_cost": 50},
    ]
    all_players_data = [
        {"code": p.code, "element_type": p.element_type, "team_code": (p.code % 15) + 1, "now_cost": 50}
        for p in squad_players
    ] + extra_players
    all_players = pd.DataFrame(all_players_data)

    data = SeasonData(
        gameweek_performances=pd.DataFrame(),
        fixtures=pd.DataFrame(),
        players=all_players,
        teams=pd.DataFrame(),
        current_gameweek=5,
        season="2024-25",
    )

    return state, data


class TestLPOptimizer:
    def test_returns_valid_lineup(self):
        state, data = _make_state_and_data()
        preds = {p.code: float(p.code) for p in state.players}
        # Also predict available players
        for code in [100, 101, 102, 103, 104]:
            preds[code] = 1.0  # low predictions, shouldn't trigger transfers
        predictions = PlayerPredictions(predictions=preds)

        optimizer = LPOptimizer()
        actions = optimizer.optimize(predictions, state, data)

        lineups = [a for a in actions if isinstance(a, SetLineup)]
        assert len(lineups) == 1
        assert len(lineups[0].starting_xi) == 11
        assert len(lineups[0].bench_order) == 4

    def test_sets_captain_and_vice(self):
        state, data = _make_state_and_data()
        preds = {p.code: float(p.code) for p in state.players}
        for code in [100, 101, 102, 103, 104]:
            preds[code] = 1.0
        predictions = PlayerPredictions(predictions=preds)

        optimizer = LPOptimizer()
        actions = optimizer.optimize(predictions, state, data)

        captains = [a for a in actions if isinstance(a, SetCaptain)]
        vices = [a for a in actions if isinstance(a, SetViceCaptain)]
        assert len(captains) == 1
        assert len(vices) == 1

    def test_captain_is_highest_predicted_starter(self):
        state, data = _make_state_and_data()
        preds = {p.code: float(p.code) for p in state.players}
        for code in [100, 101, 102, 103, 104]:
            preds[code] = 1.0
        predictions = PlayerPredictions(predictions=preds)

        optimizer = LPOptimizer()
        actions = optimizer.optimize(predictions, state, data)

        captain = [a for a in actions if isinstance(a, SetCaptain)][0]
        lineup = [a for a in actions if isinstance(a, SetLineup)][0]
        # Captain should be highest-predicted starter
        starter_preds = [(c, preds[c]) for c in lineup.starting_xi]
        best_starter = max(starter_preds, key=lambda x: x[1])[0]
        assert captain.player_id == best_starter

    def test_makes_beneficial_transfer(self):
        state, data = _make_state_and_data()
        # Squad MID code=12 predicted at 0.0, available MID code=102 predicted at 15.0
        preds = {p.code: 5.0 for p in state.players}
        preds[12] = 0.0  # worst midfielder
        for code in [100, 101, 102, 103, 104]:
            preds[code] = 1.0
        preds[102] = 15.0  # great available midfielder
        predictions = PlayerPredictions(predictions=preds)

        optimizer = LPOptimizer()
        actions = optimizer.optimize(predictions, state, data)

        transfers = [a for a in actions if isinstance(a, Transfer)]
        assert len(transfers) >= 1
        # Should transfer in player 102
        transfer_in_codes = [t.player_in for t in transfers]
        assert 102 in transfer_in_codes

    def test_no_transfer_when_not_beneficial(self):
        state, data = _make_state_and_data()
        # All squad players predicted high, available players low
        preds = {p.code: 10.0 for p in state.players}
        for code in [100, 101, 102, 103, 104]:
            preds[code] = 1.0
        predictions = PlayerPredictions(predictions=preds)

        optimizer = LPOptimizer()
        actions = optimizer.optimize(predictions, state, data)
        transfers = [a for a in actions if isinstance(a, Transfer)]
        assert len(transfers) == 0

    def test_respects_max_three_per_team(self):
        """Players from same team should be limited to 3 in squad."""
        state, data = _make_state_and_data()
        # Make all extra players same team as player 1 (team_code = 2)
        # Player 1 has team_code = (1 % 15) + 1 = 2
        data.players = data.players.copy()
        data.players.loc[data.players["code"].isin([100, 101, 102, 103, 104]), "team_code"] = 2

        preds = {p.code: 5.0 for p in state.players}
        for code in [100, 101, 102, 103, 104]:
            preds[code] = 50.0  # Very high but all same team
        predictions = PlayerPredictions(predictions=preds)

        optimizer = LPOptimizer()
        actions = optimizer.optimize(predictions, state, data)

        # Check final squad doesn't have more than 3 from team 2
        transfers = [a for a in actions if isinstance(a, Transfer)]
        out_codes = {t.player_out for t in transfers}
        in_codes = {t.player_in for t in transfers}

        squad_codes = {p.code for p in state.players}
        final_squad = (squad_codes - out_codes) | in_codes

        team_map = dict(zip(data.players["code"], data.players["team_code"]))
        team_2_count = sum(1 for c in final_squad if team_map.get(c) == 2)
        assert team_2_count <= 3

    def test_respects_budget(self):
        state, data = _make_state_and_data()
        # Make an expensive player available, but budget is tight
        state.budget = 10  # only 1.0m remaining
        data.players = data.players.copy()
        # Make available player 103 very expensive
        data.players.loc[data.players["code"] == 103, "now_cost"] = 200

        preds = {p.code: 5.0 for p in state.players}
        for code in [100, 101, 102, 103, 104]:
            preds[code] = 50.0
        predictions = PlayerPredictions(predictions=preds)

        optimizer = LPOptimizer()
        actions = optimizer.optimize(predictions, state, data)

        transfers = [a for a in actions if isinstance(a, Transfer)]
        # Should not transfer in player 103 (too expensive)
        transfer_in_codes = [t.player_in for t in transfers]
        assert 103 not in transfer_in_codes

    def test_respects_free_transfer_limit(self):
        state, data = _make_state_and_data()
        state.free_transfers = 1

        # Make multiple players much better than current squad
        preds = {p.code: 2.0 for p in state.players}
        for code in [100, 101, 102, 103, 104]:
            preds[code] = 50.0
        predictions = PlayerPredictions(predictions=preds)

        optimizer = LPOptimizer()
        actions = optimizer.optimize(predictions, state, data)

        transfers = [a for a in actions if isinstance(a, Transfer)]
        # V1: no paid hits, so at most free_transfers transfers
        assert len(transfers) <= state.free_transfers

    def test_formation_valid(self):
        """Starting XI must have valid formation."""
        state, data = _make_state_and_data()
        preds = {p.code: float(p.code) for p in state.players}
        for code in [100, 101, 102, 103, 104]:
            preds[code] = 1.0
        predictions = PlayerPredictions(predictions=preds)

        optimizer = LPOptimizer()
        actions = optimizer.optimize(predictions, state, data)

        lineup = [a for a in actions if isinstance(a, SetLineup)][0]

        # Check formation is valid
        transfers = [a for a in actions if isinstance(a, Transfer)]
        out_codes = {t.player_out for t in transfers}
        in_codes = {t.player_in for t in transfers}

        squad_codes = {p.code for p in state.players}
        final_squad = (squad_codes - out_codes) | in_codes

        # Build type map from data.players
        type_map = dict(zip(data.players["code"], data.players["element_type"]))

        xi_types = [type_map[c] for c in lineup.starting_xi]
        from collections import Counter
        counts = Counter(xi_types)
        assert counts.get(1, 0) == 1  # exactly 1 GK
        assert counts.get(2, 0) >= 3  # at least 3 DEF
        assert counts.get(3, 0) >= 2  # at least 2 MID
        assert counts.get(4, 0) >= 1  # at least 1 FWD
        assert sum(counts.values()) == 11

    def test_squad_composition_valid(self):
        """Final squad must have 2 GK, 5 DEF, 5 MID, 3 FWD."""
        state, data = _make_state_and_data()
        preds = {p.code: float(p.code) for p in state.players}
        for code in [100, 101, 102, 103, 104]:
            preds[code] = 1.0
        predictions = PlayerPredictions(predictions=preds)

        optimizer = LPOptimizer()
        actions = optimizer.optimize(predictions, state, data)

        transfers = [a for a in actions if isinstance(a, Transfer)]
        out_codes = {t.player_out for t in transfers}
        in_codes = {t.player_in for t in transfers}

        squad_codes = {p.code for p in state.players}
        final_squad = (squad_codes - out_codes) | in_codes

        type_map = dict(zip(data.players["code"], data.players["element_type"]))
        from collections import Counter
        counts = Counter(type_map[c] for c in final_squad)
        assert counts[1] == 2
        assert counts[2] == 5
        assert counts[3] == 5
        assert counts[4] == 3

    def test_fallback_on_infeasible(self):
        """If solver fails, fallback to current squad lineup."""
        state, data = _make_state_and_data()
        # Empty predictions - should still produce valid actions
        predictions = PlayerPredictions(predictions={})

        optimizer = LPOptimizer()
        actions = optimizer.optimize(predictions, state, data)

        # Should still get a lineup, captain, vice-captain
        lineups = [a for a in actions if isinstance(a, SetLineup)]
        captains = [a for a in actions if isinstance(a, SetCaptain)]
        assert len(lineups) == 1
        assert len(captains) == 1
