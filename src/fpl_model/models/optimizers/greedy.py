"""Greedy optimizer: picks the best lineup from the current squad."""

from __future__ import annotations

from fpl_model.models.base import Optimizer, PlayerPredictions, SeasonData
from fpl_model.simulation.actions import Action, SetCaptain, SetLineup, SetViceCaptain, Transfer
from fpl_model.simulation.rules import validate_formation
from fpl_model.simulation.state import PlayerInSquad, SquadState


class GreedyOptimizer(Optimizer):
    """Select the highest-predicted-points lineup from the existing squad.

    When *enable_transfers* is True, evaluates possible transfers before
    selecting the lineup. Only uses free transfers (no paid hits).
    """

    def __init__(
        self,
        enable_transfers: bool = False,
        transfer_gain_threshold: float = 1.0,
    ) -> None:
        self.enable_transfers = enable_transfers
        self.transfer_gain_threshold = transfer_gain_threshold

    def optimize(
        self,
        predictions: PlayerPredictions,
        state: SquadState,
        data: SeasonData,
    ) -> list[Action]:
        actions: list[Action] = []

        # --- Transfer evaluation (optional) ---
        if self.enable_transfers:
            transfer_actions, state = self._evaluate_transfers(predictions, state, data)
            actions.extend(transfer_actions)

        # --- Lineup / captain selection (unchanged logic) ---
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

        actions.append(SetLineup(starting_xi=starting_xi, bench_order=bench_order))
        if captain is not None:
            actions.append(SetCaptain(player_id=captain))
        if vice_captain is not None:
            actions.append(SetViceCaptain(player_id=vice_captain))

        return actions

    # ------------------------------------------------------------------
    # Transfer evaluation
    # ------------------------------------------------------------------

    def _evaluate_transfers(
        self,
        predictions: PlayerPredictions,
        state: SquadState,
        data: SeasonData,
    ) -> tuple[list[Transfer], SquadState]:
        """Greedily evaluate and execute up to *free_transfers* beneficial swaps.

        Returns the list of Transfer actions and the updated (deep-copied)
        SquadState reflecting those transfers.
        """
        from copy import deepcopy

        preds = predictions.predictions
        player_df = data.players  # must have code, element_type, team_code, now_cost

        # Work on a mutable copy so we can track budget / roster changes
        current_state = deepcopy(state)
        transfers_made: list[Transfer] = []

        for _ in range(current_state.free_transfers):
            best_gain = 0.0
            best_out: PlayerInSquad | None = None
            best_in_code: int | None = None
            best_in_cost: int | None = None
            best_in_et: int | None = None
            best_in_team: int | None = None

            squad_codes = {p.code for p in current_state.players}

            # Build team count map for the current squad
            team_counts: dict[int, int] = {}
            for p in current_state.players:
                p_row = player_df[player_df["code"] == p.code]
                if len(p_row) > 0:
                    tc = int(p_row.iloc[0]["team_code"])
                    team_counts[tc] = team_counts.get(tc, 0) + 1

            # For each squad player, find the best available swap
            for squad_player in current_state.players:
                pred_out = preds.get(squad_player.code, 0)

                # Get the squad player's team_code
                sp_row = player_df[player_df["code"] == squad_player.code]
                sp_team = int(sp_row.iloc[0]["team_code"]) if len(sp_row) > 0 else -1

                # Consider all available players of the same position
                available = player_df[
                    (player_df["element_type"] == squad_player.element_type)
                    & (~player_df["code"].isin(squad_codes))
                ]

                for _, row in available.iterrows():
                    in_code = int(row["code"])
                    in_cost = int(row["now_cost"])
                    in_team = int(row["team_code"])
                    pred_in = preds.get(in_code, 0)

                    gain = pred_in - pred_out

                    if gain <= best_gain:
                        continue

                    # Budget check: selling out_player frees sell_price
                    if current_state.budget + squad_player.sell_price < in_cost:
                        continue

                    # Max 3 per team check
                    current_team_count = team_counts.get(in_team, 0)
                    if sp_team == in_team:
                        # Replacing same-team player: count stays the same
                        pass
                    elif current_team_count >= 3:
                        continue

                    best_gain = gain
                    best_out = squad_player
                    best_in_code = in_code
                    best_in_cost = in_cost
                    best_in_et = int(row["element_type"])
                    best_in_team = in_team

            # Only make transfer if gain exceeds threshold
            if best_out is None or best_gain <= self.transfer_gain_threshold:
                break

            # Execute the transfer on our working state
            current_state.budget += best_out.sell_price
            current_state.budget -= best_in_cost
            current_state.players.remove(best_out)
            current_state.players.append(
                PlayerInSquad(
                    code=best_in_code,
                    element_type=best_in_et,
                    buy_price=best_in_cost,
                    sell_price=best_in_cost,
                )
            )

            transfers_made.append(Transfer(player_out=best_out.code, player_in=best_in_code))

        return transfers_made, current_state


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
