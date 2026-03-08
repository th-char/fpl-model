"""FPL game rules: formation validation, transfers, auto-subs, scoring."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

import pandas as pd

from fpl_model.simulation.state import PlayerInSquad, SquadState

MIN_GK = 1
MAX_GK = 1
MIN_DEF = 3
MIN_MID = 2
MIN_FWD = 1


def validate_formation(xi_types: list[int]) -> bool:
    """Check whether a list of 11 element_type ints forms a legal FPL XI."""
    counts = Counter(xi_types)
    return (
        counts.get(1, 0) == MIN_GK
        and counts.get(2, 0) >= MIN_DEF
        and counts.get(3, 0) >= MIN_MID
        and counts.get(4, 0) >= MIN_FWD
        and sum(counts.values()) == 11
    )


def calculate_transfer_cost(num_transfers: int, free_transfers: int) -> int:
    """Return the point hit for *num_transfers* given *free_transfers* banked."""
    extra = max(0, num_transfers - free_transfers)
    return extra * 4


def apply_transfers(state: SquadState, transfers: list, player_data: pd.DataFrame) -> SquadState:
    """Return a new SquadState with *transfers* applied.

    *player_data* must contain columns ``code``, ``now_cost``, ``element_type``.
    """
    new_state = deepcopy(state)
    for transfer in transfers:
        out_player = next(p for p in new_state.players if p.code == transfer.player_out)
        new_state.players.remove(out_player)
        new_state.budget += out_player.sell_price

        in_row = player_data[player_data["code"] == transfer.player_in].iloc[0]
        in_price = int(in_row["now_cost"])
        new_state.budget -= in_price
        new_state.players.append(
            PlayerInSquad(
                code=transfer.player_in,
                element_type=int(in_row["element_type"]),
                buy_price=in_price,
                sell_price=in_price,
            )
        )
    return new_state


def apply_auto_subs(
    starting_xi: list[int],
    bench_order: list[int],
    gw_data: pd.DataFrame,
    player_types: dict[int, int],
) -> tuple[list[int], list[int]]:
    """Apply FPL automatic substitution rules.

    Returns ``(final_xi, final_bench)`` after replacing non-playing starters
    with the highest-priority eligible bench player.
    """
    minutes_map = dict(zip(gw_data["player_code"], gw_data["minutes"]))
    final_xi = list(starting_xi)
    final_bench = list(bench_order)

    for starter in starting_xi:
        if minutes_map.get(starter, 0) > 0:
            continue
        # Build the types of the remaining 10 starters (without this one)
        xi_types = [player_types[p] for p in final_xi if p != starter]
        for bench_player in list(final_bench):
            if minutes_map.get(bench_player, 0) == 0:
                continue
            candidate_types = xi_types + [player_types[bench_player]]
            if validate_formation(candidate_types):
                final_xi[final_xi.index(starter)] = bench_player
                final_bench.remove(bench_player)
                break
    return final_xi, final_bench


def score_gameweek(
    starting_xi: list[int],
    bench_order: list[int],
    captain: int | None,
    vice_captain: int | None,
    gw_data: pd.DataFrame,
    active_chip: object | None = None,
) -> int:
    """Calculate total FPL points for a gameweek.

    *gw_data* must have columns ``player_code``, ``total_points``, ``minutes``.
    """
    from fpl_model.simulation.actions import ChipType

    points_map = dict(zip(gw_data["player_code"], gw_data["total_points"]))
    minutes_map = dict(zip(gw_data["player_code"], gw_data["minutes"]))

    total = sum(points_map.get(p, 0) for p in starting_xi)

    if active_chip == ChipType.BENCH_BOOST:
        total += sum(points_map.get(p, 0) for p in bench_order)

    captain_multiplier = 3 if active_chip == ChipType.TRIPLE_CAPTAIN else 2

    if captain and minutes_map.get(captain, 0) > 0:
        total += points_map.get(captain, 0) * (captain_multiplier - 1)
    elif vice_captain and minutes_map.get(vice_captain, 0) > 0:
        total += points_map.get(vice_captain, 0)

    return total


def advance_gameweek(state: SquadState, transfers_made: int) -> SquadState:
    """Return a new SquadState advanced to the next gameweek.

    Free transfers accrue (up to 5) when none are used, and reset to 1 when
    transfers are made that consume all banked free transfers.
    """
    new_state = deepcopy(state)
    new_state.current_gameweek += 1
    new_state.active_chip = None

    if transfers_made > 0:
        remaining = max(0, state.free_transfers - transfers_made)
        new_state.free_transfers = min(remaining + 1, 5)
    else:
        new_state.free_transfers = min(state.free_transfers + 1, 5)

    return new_state
