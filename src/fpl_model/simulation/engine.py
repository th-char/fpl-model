"""Season simulation engine for replaying historical FPL seasons."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

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
    update_sell_prices,
    validate_chip,
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
        players_df = self.db.read("players", where={"season": self.season})
        gw_perf_df = self.db.read("gameweek_performances", where={"season": self.season})
        fixtures_df = self.db.read("fixtures", where={"season": self.season})
        teams_df = self.db.read("teams", where={"season": self.season})

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

            # Prevent future data leakage: only show finished fixture results
            past_fixtures = fixtures_df[fixtures_df["finished"] == 1].copy()
            future_fixtures = fixtures_df[fixtures_df["finished"] != 1].copy()
            if not future_fixtures.empty:
                future_fixtures["team_h_score"] = None
                future_fixtures["team_a_score"] = None
            visible_fixtures = pd.concat([past_fixtures, future_fixtures], ignore_index=True)

            season_data = SeasonData(
                gameweek_performances=gw_perf_df[gw_perf_df["gameweek"] <= gw],
                fixtures=visible_fixtures,
                players=players_df,
                teams=teams_df,
                current_gameweek=gw,
                season=self.season,
            )

            actions = self.model.recommend(state, season_data)
            result.actions_log[gw] = actions

            # Check if Free Hit is being played this GW
            chip_actions = [a for a in actions if isinstance(a, PlayChip)]
            is_free_hit = any(a.chip_type == ChipType.FREE_HIT for a in chip_actions)

            if is_free_hit:
                state.pre_free_hit_state = deepcopy(state)

            # Process transfers
            transfers = [a for a in actions if isinstance(a, Transfer)]
            transfer_cost = calculate_transfer_cost(len(transfers), state.free_transfers)
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
                    if validate_chip(action.chip_type, state):
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

            # Update sell prices based on current market values
            state = update_sell_prices(state, gw_data)

            result.budget_history[gw] = state.budget

            # After scoring, revert Free Hit if active
            if is_free_hit and state.pre_free_hit_state:
                restored = state.pre_free_hit_state
                restored.chips_available = state.chips_available  # keep chip usage
                state = restored
                transfer_cost = 0  # Free Hit transfers are free
                result.transfer_costs[gw] = 0

            # Advance to next gameweek
            if is_free_hit:
                # Free Hit doesn't consume free transfers
                state = advance_gameweek(state, transfers_made=0)
            else:
                state = advance_gameweek(state, transfers_made=len(transfers))

        return result

    def _initial_state(
        self,
        players_df: pd.DataFrame,
        fixtures_df: pd.DataFrame,
        teams_df: pd.DataFrame,
        first_gw: int,
    ) -> SquadState:
        """Create initial squad with valid 2-5-5-3 composition from cheapest players."""
        required = {1: 2, 2: 5, 3: 5, 4: 3}  # GK, DEF, MID, FWD
        initial_players = []

        for element_type, count in required.items():
            type_players = players_df[players_df["element_type"] == element_type]
            cheapest = type_players.nsmallest(count, "now_cost")
            for _, row in cheapest.iterrows():
                initial_players.append(
                    PlayerInSquad(
                        code=int(row["code"]),
                        element_type=int(row["element_type"]),
                        buy_price=int(row["now_cost"]),
                        sell_price=int(row["now_cost"]),
                    )
                )

        state = SquadState(
            players=initial_players,
            budget=1000 - sum(p.buy_price for p in initial_players),
            free_transfers=999,
            chips_available={ct: 2 for ct in ChipType},
            current_gameweek=first_gw,
        )

        # Let the model set lineup
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
