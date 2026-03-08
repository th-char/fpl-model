"""Season simulation engine for replaying historical FPL seasons."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import pandas as pd

from fpl_model.data.db import Database
from fpl_model.models.base import ActionModel, SeasonData
from fpl_model.simulation.actions import (
    Action,
    ChipType,
    PlayChip,
    SetCaptain,
    SetLineup,
    SetViceCaptain,
    Transfer,
)
from fpl_model.simulation.rules import (
    advance_gameweek,
    apply_auto_subs,
    apply_transfers,
    calculate_transfer_cost,
    score_gameweek,
)
from fpl_model.simulation.state import PlayerInSquad, SquadState


@dataclass
class SimulationResult:
    """Results from a full season simulation."""

    total_points: int
    gameweek_points: dict[int, int]
    actions_log: dict[int, list[Action]]
    transfer_costs: dict[int, int]
    budget_history: dict[int, int]


class SeasonSimulator:
    """Replays a historical FPL season using a given ActionModel."""

    def __init__(self, model: ActionModel, season: str, db: Database) -> None:
        self.model = model
        self.season = season
        self.db = db

    def run(self) -> SimulationResult:
        """Run the full season simulation and return results."""
        players_df = self.db.read("players", where=f"season = '{self.season}'")
        gw_perf_df = self.db.read(
            "gameweek_performances", where=f"season = '{self.season}'"
        )
        fixtures_df = self.db.read("fixtures", where=f"season = '{self.season}'")
        teams_df = self.db.read("teams", where=f"season = '{self.season}'")

        gameweeks = sorted(gw_perf_df["gameweek"].unique())
        if not gameweeks:
            raise ValueError(f"No gameweek data for season {self.season}")

        state = self._initial_state(players_df, fixtures_df, teams_df, gameweeks[0])
        result = SimulationResult(
            total_points=0,
            gameweek_points={},
            actions_log={},
            transfer_costs={},
            budget_history={},
        )

        for gw in gameweeks:
            state.current_gameweek = gw

            season_data = SeasonData(
                gameweek_performances=gw_perf_df[gw_perf_df["gameweek"] <= gw],
                fixtures=fixtures_df,
                players=players_df,
                teams=teams_df,
                current_gameweek=gw,
                season=self.season,
            )

            actions = self.model.recommend(state, season_data)
            result.actions_log[gw] = actions

            # Process transfers
            transfers = [a for a in actions if isinstance(a, Transfer)]
            transfer_cost = calculate_transfer_cost(
                len(transfers), state.free_transfers
            )
            result.transfer_costs[gw] = transfer_cost

            if transfers:
                state = apply_transfers(state, transfers, players_df)

            # Process other actions
            for action in actions:
                if isinstance(action, SetCaptain):
                    state.captain = action.player_id
                elif isinstance(action, SetViceCaptain):
                    state.vice_captain = action.player_id
                elif isinstance(action, SetLineup):
                    state.starting_xi = action.starting_xi
                    state.bench_order = action.bench_order
                elif isinstance(action, PlayChip):
                    state.active_chip = action.chip_type
                    state.chips_available[action.chip_type] -= 1

            # Score the gameweek
            gw_data = gw_perf_df[gw_perf_df["gameweek"] == gw]
            player_types = {p.code: p.element_type for p in state.players}
            final_xi, final_bench = apply_auto_subs(
                state.starting_xi, state.bench_order, gw_data, player_types
            )

            gw_points = score_gameweek(
                final_xi,
                final_bench,
                state.captain,
                state.vice_captain,
                gw_data,
                state.active_chip,
            )
            gw_points -= transfer_cost

            result.gameweek_points[gw] = gw_points
            result.total_points += gw_points
            result.budget_history[gw] = state.budget

            # Advance to next gameweek
            state = advance_gameweek(state, transfers_made=len(transfers))

        return result

    def _initial_state(
        self,
        players_df: pd.DataFrame,
        fixtures_df: pd.DataFrame,
        teams_df: pd.DataFrame,
        first_gw: int,
    ) -> SquadState:
        """Create the initial squad state by picking the 15 cheapest players."""
        sorted_players = players_df.sort_values("now_cost").head(15)
        initial_players = [
            PlayerInSquad(
                code=int(row["code"]),
                element_type=int(row["element_type"]),
                buy_price=int(row["now_cost"]),
                sell_price=int(row["now_cost"]),
            )
            for _, row in sorted_players.iterrows()
        ]
        state = SquadState(
            players=initial_players,
            budget=1000 - sum(p.buy_price for p in initial_players),
            free_transfers=999,
            chips_available={ct: 2 for ct in ChipType},
            current_gameweek=first_gw,
        )

        # Let the model set the initial lineup
        season_data = SeasonData(
            gameweek_performances=pd.DataFrame(),
            fixtures=fixtures_df,
            players=players_df,
            teams=teams_df,
            current_gameweek=first_gw,
            season=self.season,
        )

        actions = self.model.recommend(state, season_data)
        for action in actions:
            if isinstance(action, SetLineup):
                state.starting_xi = action.starting_xi
                state.bench_order = action.bench_order
            elif isinstance(action, SetCaptain):
                state.captain = action.player_id
            elif isinstance(action, SetViceCaptain):
                state.vice_captain = action.player_id

        state.free_transfers = 1
        return state
