"""LP optimizer: uses PuLP to formulate an Integer Linear Program for FPL squad optimization."""

from __future__ import annotations

import logging
from collections import Counter

import pulp

from fpl_model.models.base import Optimizer, PlayerPredictions, SeasonData
from fpl_model.simulation.actions import (
    Action,
    SetCaptain,
    SetLineup,
    SetViceCaptain,
    Transfer,
)
from fpl_model.simulation.rules import validate_formation
from fpl_model.simulation.state import SquadState

logger = logging.getLogger(__name__)

# Squad composition requirements
SQUAD_COMP = {1: 2, 2: 5, 3: 5, 4: 3}  # GK, DEF, MID, FWD
XI_SIZE = 11
SQUAD_SIZE = 15


class LPOptimizer(Optimizer):
    """Optimal squad selection, transfers, lineup, and captaincy via Integer Linear Programming.

    Uses PuLP with the bundled CBC solver.  In v1, transfers are limited to
    ``free_transfers`` (no paid hits) to keep the ILP tractable.
    """

    def optimize(
        self,
        predictions: PlayerPredictions,
        state: SquadState,
        data: SeasonData,
    ) -> list[Action]:
        try:
            return self._solve(predictions, state, data)
        except Exception:
            logger.warning("LP solver failed, falling back to greedy lineup", exc_info=True)
            return self._fallback(predictions, state)

    # ------------------------------------------------------------------
    # Core ILP formulation
    # ------------------------------------------------------------------

    def _solve(
        self,
        predictions: PlayerPredictions,
        state: SquadState,
        data: SeasonData,
    ) -> list[Action]:
        preds = predictions.predictions
        if not preds:
            raise ValueError("No predictions provided")

        squad_codes = {p.code for p in state.players}
        player_df = data.players

        # Build lookup maps
        type_map: dict[int, int] = dict(zip(player_df["code"], player_df["element_type"]))
        cost_map: dict[int, int] = dict(zip(player_df["code"], player_df["now_cost"]))
        team_map: dict[int, int] = dict(zip(player_df["code"], player_df["team_code"]))
        sell_price_map: dict[int, int] = {p.code: p.sell_price for p in state.players}

        # All candidate player codes (must have predictions and be in player data)
        all_codes = [int(c) for c in player_df["code"] if c in preds]
        if not all_codes:
            raise ValueError("No candidate players with predictions")

        non_squad_codes = [c for c in all_codes if c not in squad_codes]
        current_squad_codes = [c for c in all_codes if c in squad_codes]

        free_transfers = state.free_transfers

        # ------------------------------------------------------------------
        # ILP Model
        # ------------------------------------------------------------------
        prob = pulp.LpProblem("FPL_Squad_Optimization", pulp.LpMaximize)
        solver = pulp.PULP_CBC_CMD(msg=0)

        # Decision variables
        x = {c: pulp.LpVariable(f"x_{c}", cat="Binary") for c in all_codes}  # in squad
        s = {c: pulp.LpVariable(f"s_{c}", cat="Binary") for c in all_codes}  # starts
        cap = {c: pulp.LpVariable(f"cap_{c}", cat="Binary") for c in all_codes}  # captain
        t_out = {c: pulp.LpVariable(f"tout_{c}", cat="Binary") for c in current_squad_codes}
        t_in = {c: pulp.LpVariable(f"tin_{c}", cat="Binary") for c in non_squad_codes}

        # ------------------------------------------------------------------
        # Objective: maximize predicted points for starters + captain bonus
        # ------------------------------------------------------------------
        prob += pulp.lpSum(
            preds.get(c, 0) * s[c] + preds.get(c, 0) * cap[c] for c in all_codes
        )

        # ------------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------------

        # Squad size = 15
        prob += pulp.lpSum(x[c] for c in all_codes) == SQUAD_SIZE

        # Starting XI = 11
        prob += pulp.lpSum(s[c] for c in all_codes) == XI_SIZE

        # Squad composition: 2 GK, 5 DEF, 5 MID, 3 FWD
        for pos, count in SQUAD_COMP.items():
            prob += pulp.lpSum(x[c] for c in all_codes if type_map.get(c) == pos) == count

        # Formation: exactly 1 GK in XI
        prob += pulp.lpSum(s[c] for c in all_codes if type_map.get(c) == 1) == 1

        # Formation: >= 3 DEF in XI
        prob += pulp.lpSum(s[c] for c in all_codes if type_map.get(c) == 2) >= 3

        # Formation: >= 2 MID in XI
        prob += pulp.lpSum(s[c] for c in all_codes if type_map.get(c) == 3) >= 2

        # Formation: >= 1 FWD in XI
        prob += pulp.lpSum(s[c] for c in all_codes if type_map.get(c) == 4) >= 1

        # Can only start if in squad: s[c] <= x[c]
        for c in all_codes:
            prob += s[c] <= x[c]

        # Can only captain if starting: cap[c] <= s[c]
        for c in all_codes:
            prob += cap[c] <= s[c]

        # Exactly 1 captain
        prob += pulp.lpSum(cap[c] for c in all_codes) == 1

        # Max 3 players per team
        all_teams = set(team_map.get(c) for c in all_codes if team_map.get(c) is not None)
        for team in all_teams:
            team_players = [c for c in all_codes if team_map.get(c) == team]
            prob += pulp.lpSum(x[c] for c in team_players) <= 3

        # Transfer linking for current squad: x[c] = 1 - t_out[c]
        for c in current_squad_codes:
            prob += x[c] == 1 - t_out[c]

        # Transfer linking for non-squad: x[c] = t_in[c]
        for c in non_squad_codes:
            prob += x[c] == t_in[c]

        # Limit total transfers to free_transfers (no paid hits in v1)
        prob += pulp.lpSum(t_out[c] for c in current_squad_codes) <= free_transfers
        # Number of transfers in must equal transfers out
        prob += (
            pulp.lpSum(t_in[c] for c in non_squad_codes)
            == pulp.lpSum(t_out[c] for c in current_squad_codes)
        )

        # Budget constraint:
        # sum of costs of new squad <= state.budget + sum of sell_prices of outgoing
        # Rearranged: sum(cost[c] * x[c] for new) + sum(cost[c] * x[c] for current kept)
        #           <= budget + sum(sell_price[c] * t_out[c])
        # For current squad players kept, cost is their original cost (already paid).
        # For new players, cost is now_cost.
        # Total budget available = state.budget + sell_prices of outgoing
        # Total cost of incoming = now_cost of each t_in player
        # So: sum(now_cost[c] * t_in[c]) <= state.budget + sum(sell_price[c] * t_out[c])
        prob += (
            pulp.lpSum(cost_map.get(c, 0) * t_in[c] for c in non_squad_codes)
            <= state.budget + pulp.lpSum(sell_price_map.get(c, 0) * t_out[c] for c in current_squad_codes)
        )

        # ------------------------------------------------------------------
        # Solve
        # ------------------------------------------------------------------
        status = prob.solve(solver)
        if pulp.LpStatus[status] != "Optimal":
            raise RuntimeError(f"LP solver status: {pulp.LpStatus[status]}")

        # ------------------------------------------------------------------
        # Extract solution
        # ------------------------------------------------------------------
        new_squad = {c for c in all_codes if pulp.value(x[c]) > 0.5}
        starters = {c for c in all_codes if pulp.value(s[c]) > 0.5}
        captain_code = next(c for c in all_codes if pulp.value(cap[c]) > 0.5)
        bench = new_squad - starters

        # Build transfer actions
        transferred_out = {c for c in current_squad_codes if pulp.value(t_out[c]) > 0.5}
        transferred_in = {c for c in non_squad_codes if pulp.value(t_in[c]) > 0.5}

        # Match transfers: pair out/in by position for clean Transfer objects
        transfers = self._build_transfers(transferred_out, transferred_in, type_map)

        # Order bench by predicted points descending
        bench_ordered = sorted(bench, key=lambda c: preds.get(c, 0), reverse=True)

        # Vice-captain: second-highest predicted starter (excluding captain)
        starters_by_pred = sorted(starters, key=lambda c: preds.get(c, 0), reverse=True)
        vice_captain_code = next(
            (c for c in starters_by_pred if c != captain_code),
            captain_code,
        )

        # Build actions list
        actions: list[Action] = []
        actions.extend(transfers)
        actions.append(SetLineup(starting_xi=list(starters), bench_order=bench_ordered))
        actions.append(SetCaptain(player_id=captain_code))
        actions.append(SetViceCaptain(player_id=vice_captain_code))

        return actions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_transfers(
        out_codes: set[int],
        in_codes: set[int],
        type_map: dict[int, int],
    ) -> list[Transfer]:
        """Pair outgoing and incoming players into Transfer actions.

        Pairs by position where possible for clarity, but any valid pairing works
        since all transfers are applied together.
        """
        out_list = list(out_codes)
        in_list = list(in_codes)

        if not out_list:
            return []

        # Simple pairing: match by position if possible
        transfers: list[Transfer] = []
        remaining_in = list(in_list)
        for out_code in out_list:
            out_pos = type_map.get(out_code)
            # Try to find a matching position
            match = next((c for c in remaining_in if type_map.get(c) == out_pos), None)
            if match is None and remaining_in:
                match = remaining_in[0]
            if match is not None:
                transfers.append(Transfer(player_out=out_code, player_in=match))
                remaining_in.remove(match)

        return transfers

    # ------------------------------------------------------------------
    # Fallback: greedy lineup from current squad
    # ------------------------------------------------------------------

    def _fallback(
        self,
        predictions: PlayerPredictions,
        state: SquadState,
    ) -> list[Action]:
        """When the ILP fails, fall back to a simple greedy lineup."""
        preds = predictions.predictions
        players = state.players

        # Split into GKs and outfield
        gks = sorted(
            [p for p in players if p.element_type == 1],
            key=lambda p: preds.get(p.code, 0),
            reverse=True,
        )
        outfield = sorted(
            [p for p in players if p.element_type != 1],
            key=lambda p: preds.get(p.code, 0),
            reverse=True,
        )

        starting_gk = gks[0] if gks else None

        # Guarantee formation minimums then fill
        mins = {2: 3, 3: 2, 4: 1}
        selected: list = []
        remaining = list(outfield)

        for pos, min_count in mins.items():
            pos_players = [p for p in remaining if p.element_type == pos]
            for p in pos_players[:min_count]:
                selected.append(p)
                remaining.remove(p)

        spots_left = 10 - len(selected)
        remaining_sorted = sorted(remaining, key=lambda p: preds.get(p.code, 0), reverse=True)
        selected.extend(remaining_sorted[:spots_left])

        starting_xi_players = ([starting_gk] if starting_gk else []) + selected
        starting_xi = [p.code for p in starting_xi_players]

        bench_gks = gks[1:] if len(gks) > 1 else []
        bench_outfield = [p for p in outfield if p not in selected]
        bench_ordered = bench_gks + sorted(
            bench_outfield, key=lambda p: preds.get(p.code, 0), reverse=True
        )
        bench_order = [p.code for p in bench_ordered]

        xi_by_pred = sorted(starting_xi, key=lambda c: preds.get(c, 0), reverse=True)
        captain = xi_by_pred[0] if xi_by_pred else (starting_xi[0] if starting_xi else 1)
        vice_captain = xi_by_pred[1] if len(xi_by_pred) > 1 else captain

        actions: list[Action] = [
            SetLineup(starting_xi=starting_xi, bench_order=bench_order),
            SetCaptain(player_id=captain),
            SetViceCaptain(player_id=vice_captain),
        ]
        return actions
