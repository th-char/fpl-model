"""Squad state representation for FPL simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

from fpl_model.simulation.actions import ChipType


@dataclass
class PlayerInSquad:
    """A player currently in the manager's 15-player squad."""

    code: int
    element_type: int  # 1=GK, 2=DEF, 3=MID, 4=FWD
    buy_price: int  # in 0.1m units
    sell_price: int  # in 0.1m units


@dataclass
class SquadState:
    """Complete snapshot of a manager's squad for decision-making."""

    players: list[PlayerInSquad]
    budget: int  # remaining budget in 0.1m units
    free_transfers: int
    chips_available: dict[ChipType, int]
    current_gameweek: int
    captain: int | None = None
    vice_captain: int | None = None
    starting_xi: list[int] = field(default_factory=list)
    bench_order: list[int] = field(default_factory=list)
    active_chip: ChipType | None = None
    pre_free_hit_state: SquadState | None = None
