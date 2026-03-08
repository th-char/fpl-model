"""Greedy optimizer: picks the best lineup from the current squad."""

from __future__ import annotations

from fpl_model.models.base import Optimizer, PlayerPredictions, SeasonData
from fpl_model.simulation.actions import Action, SetCaptain, SetLineup, SetViceCaptain
from fpl_model.simulation.rules import validate_formation
from fpl_model.simulation.state import SquadState


class GreedyOptimizer(Optimizer):
    """Select the highest-predicted-points lineup from the existing squad.

    Does not make transfers -- only sets lineup, captain, and vice-captain.
    """

    def optimize(
        self,
        predictions: PlayerPredictions,
        state: SquadState,
        data: SeasonData,
    ) -> list[Action]:
        preds = predictions.predictions
        players = state.players

        # Split into GKs and outfield
        gks = [p for p in players if p.element_type == 1]
        outfield = [p for p in players if p.element_type != 1]

        # Pick starting GK (highest predicted)
        gks_sorted = sorted(gks, key=lambda p: preds.get(p.code, 0), reverse=True)
        starting_gk = gks_sorted[0] if gks_sorted else None

        # Sort outfield by predicted points descending
        outfield_sorted = sorted(outfield, key=lambda p: preds.get(p.code, 0), reverse=True)

        # Greedily select 10 outfield starters respecting formation minimums
        starting_outfield = _select_outfield(outfield_sorted, preds)

        # Build starting XI and bench
        starting_xi_players = ([starting_gk] if starting_gk else []) + starting_outfield
        starting_xi = [p.code for p in starting_xi_players]

        bench_gks = [p for p in gks_sorted[1:]] if len(gks_sorted) > 1 else []
        bench_outfield = [p for p in outfield_sorted if p not in starting_outfield]
        # Bench ordered by predicted points descending (GK substitute first per FPL convention)
        bench_ordered = bench_gks + sorted(bench_outfield, key=lambda p: preds.get(p.code, 0), reverse=True)
        bench_order = [p.code for p in bench_ordered]

        # Captain = highest predicted in starting XI; vice = second highest
        xi_by_pred = sorted(starting_xi, key=lambda c: preds.get(c, 0), reverse=True)
        captain = xi_by_pred[0] if xi_by_pred else None
        vice_captain = xi_by_pred[1] if len(xi_by_pred) > 1 else captain

        actions: list[Action] = [SetLineup(starting_xi=starting_xi, bench_order=bench_order)]
        if captain is not None:
            actions.append(SetCaptain(player_id=captain))
        if vice_captain is not None:
            actions.append(SetViceCaptain(player_id=vice_captain))

        return actions


def _select_outfield(outfield_sorted, preds) -> list:
    """Greedily pick 10 outfield players respecting formation constraints.

    Minimums: >=3 DEF, >=2 MID, >=1 FWD among 10 outfield starters.
    Strategy: first guarantee minimums, then fill remaining slots by
    predicted points.
    """
    from collections import Counter

    # Track how many of each position we need at minimum
    mins = {2: 3, 3: 2, 4: 1}  # DEF, MID, FWD
    total_needed = 10

    # First pass: ensure minimums are met
    selected: list = []
    remaining = list(outfield_sorted)

    # Guarantee minimum for each position
    for pos, min_count in mins.items():
        pos_players = [p for p in remaining if p.element_type == pos]
        for p in pos_players[:min_count]:
            selected.append(p)
            remaining.remove(p)

    # Fill remaining slots with highest predicted from remaining outfield
    spots_left = total_needed - len(selected)
    remaining_sorted = sorted(remaining, key=lambda p: preds.get(p.code, 0), reverse=True)
    for p in remaining_sorted[:spots_left]:
        selected.append(p)

    # Verify formation
    xi_types = [1] + [p.element_type for p in selected]
    if not validate_formation(xi_types):
        # Fallback: just take top 10 outfield (should rarely happen with 5-5-3 squad)
        selected = list(outfield_sorted[:total_needed])

    return selected
