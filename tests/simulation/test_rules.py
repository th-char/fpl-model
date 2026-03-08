# tests/simulation/test_rules.py
import pandas as pd

from fpl_model.simulation.actions import (
    ChipType,
)
from fpl_model.simulation.rules import (
    advance_gameweek,
    apply_auto_subs,
    calculate_transfer_cost,
    validate_formation,
)
from fpl_model.simulation.state import PlayerInSquad, SquadState


def make_squad(gk=2, defs=5, mids=5, fwds=3, budget=0):
    """Helper to create a basic squad state."""
    players = []
    code = 1
    for _ in range(gk):
        players.append(PlayerInSquad(code=code, element_type=1, buy_price=45, sell_price=45))
        code += 1
    for _ in range(defs):
        players.append(PlayerInSquad(code=code, element_type=2, buy_price=50, sell_price=50))
        code += 1
    for _ in range(mids):
        players.append(PlayerInSquad(code=code, element_type=3, buy_price=60, sell_price=60))
        code += 1
    for _ in range(fwds):
        players.append(PlayerInSquad(code=code, element_type=4, buy_price=70, sell_price=70))
        code += 1

    # Default formation: 1 GK, 4 DEF, 4 MID, 2 FWD = 11 starters
    # Starting XI: GK1, DEF3-6, MID8-11, FWD13-14
    starting_xi = [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]
    bench_order = [2, 7, 12, 15]  # GK2, DEF7, MID12, FWD15

    return SquadState(
        players=players,
        budget=budget,
        free_transfers=1,
        chips_available={ct: 2 for ct in ChipType},
        current_gameweek=1,
        captain=8,  # a midfielder
        vice_captain=13,  # a forward
        starting_xi=starting_xi,
        bench_order=bench_order,
    )


class TestValidateFormation:
    def test_valid_442(self):
        xi_types = [1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4]
        assert validate_formation(xi_types) is True

    def test_valid_343(self):
        xi_types = [1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4]
        assert validate_formation(xi_types) is True

    def test_invalid_too_few_defenders(self):
        xi_types = [1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4]
        assert validate_formation(xi_types) is False

    def test_invalid_no_forward(self):
        xi_types = [1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
        assert validate_formation(xi_types) is False


class TestTransferCost:
    def test_one_free_transfer_no_cost(self):
        assert calculate_transfer_cost(num_transfers=1, free_transfers=1) == 0

    def test_extra_transfer_costs_4(self):
        assert calculate_transfer_cost(num_transfers=2, free_transfers=1) == 4

    def test_multiple_extras(self):
        assert calculate_transfer_cost(num_transfers=4, free_transfers=2) == 8

    def test_zero_transfers(self):
        assert calculate_transfer_cost(num_transfers=0, free_transfers=1) == 0


class TestAutoSubs:
    def test_auto_sub_replaces_non_playing_starter(self):
        squad = make_squad()
        # Player code=3 (a defender in starting XI) didn't play
        gw_data = pd.DataFrame(
            {
                "player_code": [p.code for p in squad.players],
                "minutes": [90 if p.code != 3 else 0 for p in squad.players],
                "total_points": [5 if p.code != 3 else 0 for p in squad.players],
            }
        )
        player_types = {p.code: p.element_type for p in squad.players}
        final_xi, final_bench = apply_auto_subs(
            squad.starting_xi, squad.bench_order, gw_data, player_types
        )
        assert 3 not in final_xi
        # A bench player should have come in
        assert any(b_code in final_xi for b_code in squad.bench_order)

    def test_no_sub_if_all_played(self):
        squad = make_squad()
        gw_data = pd.DataFrame(
            {
                "player_code": [p.code for p in squad.players],
                "minutes": [90] * 15,
                "total_points": [5] * 15,
            }
        )
        player_types = {p.code: p.element_type for p in squad.players}
        final_xi, final_bench = apply_auto_subs(
            squad.starting_xi, squad.bench_order, gw_data, player_types
        )
        assert final_xi == squad.starting_xi


class TestAdvanceGameweek:
    def test_accrues_free_transfer(self):
        squad = make_squad()
        squad.free_transfers = 1
        new_state = advance_gameweek(squad, transfers_made=0)
        assert new_state.free_transfers == 2

    def test_caps_free_transfers_at_5(self):
        squad = make_squad()
        squad.free_transfers = 5
        new_state = advance_gameweek(squad, transfers_made=0)
        assert new_state.free_transfers == 5

    def test_resets_to_1_if_transfers_made(self):
        squad = make_squad()
        squad.free_transfers = 2
        new_state = advance_gameweek(squad, transfers_made=2)
        assert new_state.free_transfers == 1
