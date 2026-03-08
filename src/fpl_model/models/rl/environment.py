"""RL environment wrapping FPL season simulation.

Provides a gym-like interface for training RL agents on historical FPL data.
The environment steps through gameweeks, scoring each using actual historical
performance data and the simulation rules engine.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from fpl_model.data.db import Database
from fpl_model.models.base import SeasonData
from fpl_model.models.features import build_player_features
from fpl_model.simulation.actions import ChipType, Transfer
from fpl_model.simulation.rules import (
    advance_gameweek,
    apply_auto_subs,
    apply_transfers,
    score_gameweek,
    validate_formation,
)
from fpl_model.simulation.state import PlayerInSquad, SquadState

# Per-player feature count: 4 (element_type one-hot) + 1 (price) + 1 (form_3)
# + 1 (form_5) + 1 (minutes_rolling) + 1 (bps_rolling) + 1 (is_starter) = 10
_PLAYER_FEATURES = 10
_NUM_SQUAD_PLAYERS = 15
_GLOBAL_FEATURES = 7  # budget, free_transfers, gw, 4 chip counts
STATE_DIM = _NUM_SQUAD_PLAYERS * _PLAYER_FEATURES + _GLOBAL_FEATURES


class FPLEnvironment:
    """Gym-like RL environment for FPL season simulation.

    Wraps the simulation rules to provide a step-based interface where
    each step corresponds to one gameweek.

    Parameters
    ----------
    season : str
        Season identifier (e.g. "2024-25").
    db : Database
        Database instance with season data loaded.
    """

    def __init__(self, season: str, db: Database) -> None:
        self.season = season
        self.db = db

        # Load all season data from DB
        self._players_df = db.read("players", where={"season": season})
        self._gw_perf_df = db.read("gameweek_performances", where={"season": season})
        self._fixtures_df = db.read("fixtures", where={"season": season})
        self._teams_df = db.read("teams", where={"season": season})

        # Determine available gameweeks
        self._all_gws = sorted(self._gw_perf_df["gameweek"].unique())
        self._max_gw = max(self._all_gws) if self._all_gws else 0

        self._state: SquadState | None = None
        self._current_gw_idx: int = 0

    @property
    def state_dim(self) -> int:
        """Return the dimensionality of the state vector."""
        return STATE_DIM

    def reset(self) -> np.ndarray:
        """Reset the environment to the start of the season.

        Creates an initial squad of the cheapest players per position
        (2 GK, 5 DEF, 5 MID, 3 FWD) and sets a default lineup.

        Returns
        -------
        np.ndarray
            Initial state vector of shape ``(state_dim,)``.
        """
        self._current_gw_idx = 0
        squad_players = self._build_initial_squad()

        # Set default lineup: 1 GK + 4 DEF + 4 MID + 2 FWD = 11
        gks = [p for p in squad_players if p.element_type == 1]
        defs = [p for p in squad_players if p.element_type == 2]
        mids = [p for p in squad_players if p.element_type == 3]
        fwds = [p for p in squad_players if p.element_type == 4]

        starting_xi_players = gks[:1] + defs[:4] + mids[:4] + fwds[:2]
        bench_players = gks[1:] + defs[4:] + mids[4:] + fwds[2:]

        starting_xi = [p.code for p in starting_xi_players]
        bench_order = [p.code for p in bench_players]

        # Captain and vice captain from starting midfielders/forwards
        captain = starting_xi_players[-1].code  # last forward
        vice_captain = starting_xi_players[-2].code  # second-to-last

        total_cost = sum(p.buy_price for p in squad_players)
        budget = 1000 - total_cost  # 1000 = 100.0m in 0.1m units

        self._state = SquadState(
            players=squad_players,
            budget=budget,
            free_transfers=1,
            chips_available={ct: 2 for ct in ChipType},
            current_gameweek=self._all_gws[0] if self._all_gws else 1,
            captain=captain,
            vice_captain=vice_captain,
            starting_xi=starting_xi,
            bench_order=bench_order,
        )

        return self._encode_state()

    def step(
        self, action_dict: dict
    ) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one gameweek step.

        Parameters
        ----------
        action_dict : dict
            Action dictionary. Use ``null_action()`` for a no-op.
            Keys: ``"transfers"`` (list of (out, in) tuples),
            ``"starting_xi"`` (list of codes), ``"bench_order"`` (list of codes),
            ``"captain"`` (int), ``"vice_captain"`` (int), ``"chip"`` (ChipType or None).

        Returns
        -------
        tuple
            ``(state, reward, done, info)`` where state is the new state vector,
            reward is GW points as float, done indicates season end, and info
            contains ``{"gw": int, "points": int}``.
        """
        assert self._state is not None, "Must call reset() before step()"

        current_gw = self._state.current_gameweek

        # Apply transfers if any
        transfers = action_dict.get("transfers", [])
        transfer_objects = [Transfer(player_out=t[0], player_in=t[1]) for t in transfers]
        num_transfers = len(transfer_objects)

        if transfer_objects:
            self._state = apply_transfers(self._state, transfer_objects, self._players_df)

        # Apply lineup changes if provided
        if "starting_xi" in action_dict and action_dict["starting_xi"]:
            self._state.starting_xi = action_dict["starting_xi"]
        if "bench_order" in action_dict and action_dict["bench_order"]:
            self._state.bench_order = action_dict["bench_order"]
        if "captain" in action_dict and action_dict["captain"] is not None:
            self._state.captain = action_dict["captain"]
        if "vice_captain" in action_dict and action_dict["vice_captain"] is not None:
            self._state.vice_captain = action_dict["vice_captain"]

        # Apply chip if specified
        chip = action_dict.get("chip", None)
        if chip is not None:
            self._state.active_chip = chip

        # Get GW performance data
        gw_data = self._gw_perf_df[self._gw_perf_df["gameweek"] == current_gw].copy()
        if "player_code" not in gw_data.columns:
            gw_data = gw_data.rename(columns={"code": "player_code"})

        # Build player_types map for auto-subs
        player_types = {p.code: p.element_type for p in self._state.players}

        # Apply auto-subs
        final_xi, final_bench = apply_auto_subs(
            self._state.starting_xi,
            self._state.bench_order,
            gw_data,
            player_types,
        )

        # Score the gameweek
        points = score_gameweek(
            starting_xi=final_xi,
            bench_order=final_bench,
            captain=self._state.captain,
            vice_captain=self._state.vice_captain,
            gw_data=gw_data,
            active_chip=self._state.active_chip,
        )

        # Advance to next gameweek
        self._state = advance_gameweek(self._state, num_transfers)
        self._current_gw_idx += 1

        # Check if season is done
        done = self._current_gw_idx >= len(self._all_gws)

        # Update current_gameweek in state for encoding
        if not done:
            self._state.current_gameweek = self._all_gws[self._current_gw_idx]

        reward = float(points)
        info = {"gw": current_gw, "points": points}

        return self._encode_state(), reward, done, info

    def null_action(self) -> dict:
        """Return a no-op action dict (keep current lineup, no transfers).

        Returns
        -------
        dict
            Action dict with empty transfers and current lineup preserved.
        """
        return {
            "transfers": [],
            "starting_xi": [],
            "bench_order": [],
            "captain": None,
            "vice_captain": None,
            "chip": None,
        }

    def _encode_state(self) -> np.ndarray:
        """Encode the current SquadState into a fixed-size float vector.

        State vector layout (~157 dims):
        - Per squad player (15 x 10): element_type one-hot (4), normalized price,
          form_3, form_5, minutes_rolling, bps_rolling, is_starter (1/0)
        - Global (7): budget/1000, free_transfers/5, current_gw/38,
          4 chip remaining counts

        Returns
        -------
        np.ndarray
            Float32 vector of shape ``(state_dim,)``.
        """
        state = self._state
        if state is None:
            return np.zeros(self.state_dim, dtype=np.float32)

        vec = np.zeros(self.state_dim, dtype=np.float32)

        # Build SeasonData for feature computation (only past data)
        current_gw = state.current_gameweek
        past_perf = self._gw_perf_df[self._gw_perf_df["gameweek"] < current_gw]
        season_data = SeasonData(
            gameweek_performances=past_perf,
            fixtures=self._fixtures_df,
            players=self._players_df,
            teams=self._teams_df,
            current_gameweek=current_gw,
            season=self.season,
        )

        xi_set = set(state.starting_xi)

        # Encode each player in the squad
        for i, player in enumerate(state.players[:_NUM_SQUAD_PLAYERS]):
            offset = i * _PLAYER_FEATURES

            # Element type one-hot (4 dims)
            if 1 <= player.element_type <= 4:
                vec[offset + player.element_type - 1] = 1.0

            # Normalized price (now_cost / 200, typical range 40-150 -> 0.2-0.75)
            vec[offset + 4] = player.sell_price / 200.0

            # Get rolling features from feature module
            features = build_player_features(season_data, player.code, current_gw)
            vec[offset + 5] = features.get("form_3", 0.0) / 10.0  # normalize
            vec[offset + 6] = features.get("form_5", 0.0) / 10.0
            vec[offset + 7] = features.get("minutes_rolling", 0.0) / 90.0
            vec[offset + 8] = features.get("bps_rolling", 0.0) / 50.0

            # Is starter
            vec[offset + 9] = 1.0 if player.code in xi_set else 0.0

        # Global features
        global_offset = _NUM_SQUAD_PLAYERS * _PLAYER_FEATURES
        vec[global_offset + 0] = state.budget / 1000.0
        vec[global_offset + 1] = state.free_transfers / 5.0
        vec[global_offset + 2] = current_gw / 38.0
        vec[global_offset + 3] = state.chips_available.get(ChipType.WILDCARD, 0) / 2.0
        vec[global_offset + 4] = state.chips_available.get(ChipType.FREE_HIT, 0) / 2.0
        vec[global_offset + 5] = state.chips_available.get(ChipType.BENCH_BOOST, 0) / 2.0
        vec[global_offset + 6] = state.chips_available.get(ChipType.TRIPLE_CAPTAIN, 0) / 2.0

        return vec

    def _build_initial_squad(self) -> list[PlayerInSquad]:
        """Build the cheapest valid squad: 2 GK, 5 DEF, 5 MID, 3 FWD.

        Returns
        -------
        list[PlayerInSquad]
            List of 15 PlayerInSquad objects.
        """
        players_df = self._players_df.copy()
        squad: list[PlayerInSquad] = []

        position_needs = {1: 2, 2: 5, 3: 5, 4: 3}

        for et, count in position_needs.items():
            pos_players = players_df[players_df["element_type"] == et].sort_values("now_cost")
            selected = pos_players.head(count)
            for _, row in selected.iterrows():
                cost = int(row["now_cost"])
                squad.append(
                    PlayerInSquad(
                        code=int(row["code"]),
                        element_type=et,
                        buy_price=cost,
                        sell_price=cost,
                    )
                )

        return squad
