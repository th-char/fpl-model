"""FPL manager actions: transfers, captaincy, lineup, and chips."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ChipType(Enum):
    """Chips available in FPL."""

    WILDCARD = "wildcard"
    FREE_HIT = "free_hit"
    BENCH_BOOST = "bench_boost"
    TRIPLE_CAPTAIN = "triple_captain"


@dataclass(frozen=True)
class Transfer:
    """A single player transfer: sell player_out, buy player_in."""

    player_out: int
    player_in: int


@dataclass(frozen=True)
class SetCaptain:
    """Designate a player as captain (scores double)."""

    player_id: int


@dataclass(frozen=True)
class SetViceCaptain:
    """Designate a player as vice-captain."""

    player_id: int


@dataclass(frozen=True)
class SetLineup:
    """Set the starting XI and bench order."""

    starting_xi: list[int]
    bench_order: list[int]


@dataclass(frozen=True)
class PlayChip:
    """Activate a chip for the gameweek."""

    chip_type: ChipType


Action = Transfer | SetCaptain | SetViceCaptain | SetLineup | PlayChip
