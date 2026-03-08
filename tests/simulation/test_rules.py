# tests/simulation/test_rules.py
import pandas as pd
import pytest

from fpl_model.simulation.actions import (
    ChipType,
    Transfer,
)
from fpl_model.simulation.rules import (
    advance_gameweek,
    apply_auto_subs,
    apply_transfers,
    calculate_transfer_cost,
    score_gameweek,
    update_sell_prices,
    validate_chip,
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


class TestValidateChip:
    def test_valid_chip(self):
        squad = make_squad()
        squad.current_gameweek = 2
        assert validate_chip(ChipType.BENCH_BOOST, squad) is True

    def test_no_uses_remaining(self):
        squad = make_squad()
        squad.chips_available[ChipType.BENCH_BOOST] = 0
        assert validate_chip(ChipType.BENCH_BOOST, squad) is False

    def test_already_active_chip(self):
        squad = make_squad()
        squad.active_chip = ChipType.WILDCARD
        assert validate_chip(ChipType.BENCH_BOOST, squad) is False

    def test_free_hit_gw1_blocked(self):
        squad = make_squad()
        squad.current_gameweek = 1
        assert validate_chip(ChipType.FREE_HIT, squad) is False

    def test_wildcard_gw1_blocked(self):
        squad = make_squad()
        squad.current_gameweek = 1
        assert validate_chip(ChipType.WILDCARD, squad) is False

    def test_free_hit_allowed_after_gw1(self):
        squad = make_squad()
        squad.current_gameweek = 5
        assert validate_chip(ChipType.FREE_HIT, squad) is True


class TestApplyTransfers:
    def test_basic_transfer(self):
        squad = make_squad(budget=100)
        players_df = pd.DataFrame({
            "code": [99],
            "now_cost": [50],
            "element_type": [2],
        })
        transfer = Transfer(player_out=3, player_in=99)
        new_state = apply_transfers(squad, [transfer], players_df)
        codes = [p.code for p in new_state.players]
        assert 3 not in codes
        assert 99 in codes

    def test_budget_updated(self):
        squad = make_squad(budget=100)
        players_df = pd.DataFrame({
            "code": [99],
            "now_cost": [60],
            "element_type": [2],
        })
        transfer = Transfer(player_out=3, player_in=99)
        new_state = apply_transfers(squad, [transfer], players_df)
        # sold player at sell_price=50, bought at 60: budget = 100 + 50 - 60 = 90
        assert new_state.budget == 90

    def test_insufficient_budget_raises(self):
        squad = make_squad(budget=0)
        players_df = pd.DataFrame({
            "code": [99],
            "now_cost": [60],
            "element_type": [2],
        })
        transfer = Transfer(player_out=3, player_in=99)
        with pytest.raises(ValueError, match="Insufficient budget"):
            apply_transfers(squad, [transfer], players_df)


class TestScoreGameweek:
    def test_basic_scoring(self):
        squad = make_squad()
        gw_data = pd.DataFrame({
            "player_code": list(range(1, 16)),
            "total_points": [5] * 15,
            "minutes": [90] * 15,
        })
        points = score_gameweek(
            squad.starting_xi, squad.bench_order,
            squad.captain, squad.vice_captain, gw_data,
        )
        # 11 starters * 5 = 55, captain (8) gets +5 (doubled) = 60
        assert points == 60

    def test_bench_boost(self):
        squad = make_squad()
        gw_data = pd.DataFrame({
            "player_code": list(range(1, 16)),
            "total_points": [5] * 15,
            "minutes": [90] * 15,
        })
        points = score_gameweek(
            squad.starting_xi, squad.bench_order,
            squad.captain, squad.vice_captain, gw_data,
            active_chip=ChipType.BENCH_BOOST,
        )
        # 11*5 + 4*5 = 75, captain +5 = 80
        assert points == 80

    def test_triple_captain(self):
        squad = make_squad()
        gw_data = pd.DataFrame({
            "player_code": list(range(1, 16)),
            "total_points": [5] * 15,
            "minutes": [90] * 15,
        })
        points = score_gameweek(
            squad.starting_xi, squad.bench_order,
            squad.captain, squad.vice_captain, gw_data,
            active_chip=ChipType.TRIPLE_CAPTAIN,
        )
        # 11*5 = 55, captain gets triple so +10 = 65
        assert points == 65

    def test_vice_captain_when_captain_doesnt_play(self):
        squad = make_squad()
        minutes = [90] * 15
        minutes[7] = 0  # captain (code=8, index 7) doesn't play
        gw_data = pd.DataFrame({
            "player_code": list(range(1, 16)),
            "total_points": [5] * 15,
            "minutes": minutes,
        })
        points = score_gameweek(
            squad.starting_xi, squad.bench_order,
            squad.captain, squad.vice_captain, gw_data,
        )
        # Starting XI sum = 11*5 = 55 (captain's 5 pts still counted in starting_xi sum)
        # Captain didn't play -> vice_captain bonus = +5
        assert points == 60


class TestUpdateSellPrices:
    def test_price_increase(self):
        squad = make_squad()
        gw_data = pd.DataFrame({
            "player_code": [3],
            "value": [54],
        })
        new_state = update_sell_prices(squad, gw_data)
        player3 = next(p for p in new_state.players if p.code == 3)
        # profit = (54 - 50) // 2 = 2, sell_price = 50 + 2 = 52
        assert player3.sell_price == 52

    def test_price_decrease(self):
        squad = make_squad()
        gw_data = pd.DataFrame({
            "player_code": [3],
            "value": [45],
        })
        new_state = update_sell_prices(squad, gw_data)
        player3 = next(p for p in new_state.players if p.code == 3)
        assert player3.sell_price == 45

    def test_no_change(self):
        squad = make_squad()
        gw_data = pd.DataFrame({
            "player_code": [3],
            "value": [50],
        })
        new_state = update_sell_prices(squad, gw_data)
        player3 = next(p for p in new_state.players if p.code == 3)
        assert player3.sell_price == 50
