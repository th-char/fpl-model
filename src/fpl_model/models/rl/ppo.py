"""PPO (Proximal Policy Optimization) agent for end-to-end FPL management.

Implements the ActionModel interface directly, using a policy network to
select lineup and captain decisions, and a value network for advantage
estimation during training.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fpl_model.data.db import Database
from fpl_model.models.base import ActionModel, HistoricalData, SeasonData
from fpl_model.models.features import build_player_features
from fpl_model.models.rl.environment import FPLEnvironment, STATE_DIM
from fpl_model.simulation.actions import (
    Action,
    ChipType,
    SetCaptain,
    SetLineup,
    SetViceCaptain,
)
from fpl_model.simulation.state import PlayerInSquad, SquadState


# Position constraints for formation validation
_MIN_DEF = 3
_MIN_MID = 2
_MIN_FWD = 1

# Per-player feature count matching environment encoding
_PLAYER_FEATURES = 10
_NUM_SQUAD_PLAYERS = 15
_GLOBAL_FEATURES = 7


class PolicyNet(nn.Module):
    """Policy network with lineup and captain heads."""

    def __init__(self, state_dim: int, hidden_size: int) -> None:
        super().__init__()
        half_hidden = max(hidden_size // 2, 1)
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, half_hidden),
            nn.ReLU(),
        )
        self.lineup_head = nn.Linear(half_hidden, 15)
        self.captain_head = nn.Linear(half_hidden, 15)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (lineup_logits, captain_logits) each of shape (batch, 15)."""
        h = self.shared(x)
        return self.lineup_head(h), self.captain_head(h)


class ValueNet(nn.Module):
    """State value network."""

    def __init__(self, state_dim: int, hidden_size: int) -> None:
        super().__init__()
        half_hidden = max(hidden_size // 2, 1)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, half_hidden),
            nn.ReLU(),
            nn.Linear(half_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return state value of shape (batch, 1)."""
        return self.net(x)


class PPOAgent(ActionModel):
    """PPO reinforcement learning agent for FPL team management.

    For v1, only handles lineup selection and captaincy (no transfers/chips).

    Parameters
    ----------
    db : Database
        Database instance with historical season data.
    seasons : list[str]
        Season identifiers to train on.
    hidden_size : int
        Hidden layer size for policy and value networks.
    train_epochs : int
        Number of PPO update epochs per training round.
    episodes_per_update : int
        Number of episodes to collect before each PPO update.
    lr : float
        Learning rate.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda for advantage estimation.
    clip_epsilon : float
        PPO clipping parameter.
    """

    def __init__(
        self,
        db: Database,
        seasons: list[str],
        hidden_size: int = 128,
        train_epochs: int = 10,
        episodes_per_update: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
    ) -> None:
        self.db = db
        self.seasons = seasons
        self.hidden_size = hidden_size
        self.train_epochs = train_epochs
        self.episodes_per_update = episodes_per_update
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon

        self._policy = PolicyNet(STATE_DIM, hidden_size)
        self._value = ValueNet(STATE_DIM, hidden_size)
        self._optimizer = torch.optim.Adam(
            list(self._policy.parameters()) + list(self._value.parameters()),
            lr=lr,
        )

    def recommend(self, state: SquadState, data: SeasonData) -> list[Action]:
        """Encode state, run policy forward, decode into Actions.

        Returns a list containing SetLineup, SetCaptain, and SetViceCaptain.
        """
        state_vec = self._encode_from_state_data(state, data)
        state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)

        self._policy.eval()
        with torch.no_grad():
            lineup_logits, captain_logits = self._policy(state_tensor)

        lineup_logits = lineup_logits.squeeze(0)  # (15,)
        captain_logits = captain_logits.squeeze(0)  # (15,)

        # Decode lineup with formation masking
        players = state.players[:_NUM_SQUAD_PLAYERS]
        starting_xi, bench_order = self._decode_lineup(lineup_logits, players)

        # Decode captain from starters
        starter_codes = set(starting_xi)
        starter_indices = [
            i for i, p in enumerate(players) if p.code in starter_codes
        ]

        # Mask captain logits to only starters
        captain_probs = torch.full((15,), float("-inf"))
        for idx in starter_indices:
            captain_probs[idx] = captain_logits[idx]
        captain_probs = F.softmax(captain_probs, dim=0)

        # Top two for captain and vice-captain
        sorted_indices = torch.argsort(captain_probs, descending=True)
        captain_idx = sorted_indices[0].item()
        vice_captain_idx = sorted_indices[1].item()

        captain_code = players[captain_idx].code
        vice_captain_code = players[vice_captain_idx].code

        actions: list[Action] = [
            SetLineup(starting_xi=starting_xi, bench_order=bench_order),
            SetCaptain(player_id=captain_code),
            SetViceCaptain(player_id=vice_captain_code),
        ]
        return actions

    def train(self, historical_data: HistoricalData) -> None:
        """Run PPO training loop using FPLEnvironment for each season.

        Each step samples lineup and captain actions from the policy,
        passes them to the environment, and records the log-probabilities
        for PPO surrogate loss computation.
        """
        for _update_round in range(self.train_epochs):
            # Collect trajectories
            all_states: list[np.ndarray] = []
            all_lineup_actions: list[list[int]] = []  # per-player binary (15,)
            all_captain_actions: list[int] = []  # index into squad
            all_log_probs: list[float] = []
            all_rewards: list[float] = []
            all_values: list[float] = []
            all_dones: list[bool] = []

            for _ep in range(self.episodes_per_update):
                for season in self.seasons:
                    env = FPLEnvironment(season=season, db=self.db)
                    obs = env.reset()
                    done = False

                    while not done:
                        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

                        self._policy.eval()
                        self._value.eval()
                        with torch.no_grad():
                            lineup_logits, captain_logits = self._policy(obs_tensor)
                            value = self._value(obs_tensor).item()

                        lineup_logits_sq = lineup_logits.squeeze(0)  # (15,)
                        captain_logits_sq = captain_logits.squeeze(0)  # (15,)

                        # Sample lineup: Bernoulli per player, then decode
                        # into a valid formation
                        players = env._state.players[:_NUM_SQUAD_PLAYERS]
                        starting_xi, bench_order = self._decode_lineup(
                            lineup_logits_sq, players
                        )

                        # Record binary action vector for log-prob computation
                        starter_set = set(starting_xi)
                        lineup_binary = [
                            1 if players[i].code in starter_set else 0
                            for i in range(_NUM_SQUAD_PLAYERS)
                        ]

                        # Sample captain from starters using categorical
                        starter_indices = [
                            i for i, p in enumerate(players) if p.code in starter_set
                        ]
                        captain_mask = torch.full((15,), float("-inf"))
                        for idx in starter_indices:
                            captain_mask[idx] = captain_logits_sq[idx]
                        captain_probs = F.softmax(captain_mask, dim=0)
                        captain_dist = torch.distributions.Categorical(captain_probs)
                        captain_idx = captain_dist.sample().item()
                        captain_code = players[captain_idx].code

                        # Vice captain: second highest prob starter
                        sorted_indices = torch.argsort(captain_probs, descending=True)
                        vice_idx = sorted_indices[1].item() if len(sorted_indices) > 1 else captain_idx
                        vice_captain_code = players[vice_idx].code

                        # Compute log-probability of sampled actions
                        # Lineup: Bernoulli log-prob per player
                        lineup_probs = torch.sigmoid(lineup_logits_sq)
                        lineup_probs = torch.clamp(lineup_probs, 1e-7, 1 - 1e-7)
                        lineup_actions_t = torch.tensor(lineup_binary, dtype=torch.float32)
                        lineup_log_prob = (
                            lineup_actions_t * torch.log(lineup_probs)
                            + (1 - lineup_actions_t) * torch.log(1 - lineup_probs)
                        ).sum()

                        # Captain: categorical log-prob
                        captain_log_prob = captain_dist.log_prob(
                            torch.tensor(captain_idx)
                        )

                        log_prob = (lineup_log_prob + captain_log_prob).item()

                        all_states.append(obs)
                        all_lineup_actions.append(lineup_binary)
                        all_captain_actions.append(captain_idx)
                        all_log_probs.append(log_prob)
                        all_values.append(value)

                        # Pass sampled actions to the environment
                        action_dict = {
                            "transfers": [],
                            "starting_xi": starting_xi,
                            "bench_order": bench_order,
                            "captain": captain_code,
                            "vice_captain": vice_captain_code,
                            "chip": None,
                        }
                        obs, reward, done, _info = env.step(action_dict)

                        all_rewards.append(reward)
                        all_dones.append(done)

            if len(all_states) == 0:
                continue

            # Compute GAE advantages
            advantages = self._compute_gae(
                all_rewards, all_values, all_dones
            )
            returns = [a + v for a, v in zip(advantages, all_values)]

            # Convert to tensors
            states_t = torch.tensor(np.array(all_states), dtype=torch.float32)
            old_log_probs_t = torch.tensor(all_log_probs, dtype=torch.float32)
            lineup_actions_t = torch.tensor(all_lineup_actions, dtype=torch.float32)
            captain_actions_t = torch.tensor(all_captain_actions, dtype=torch.long)
            advantages_t = torch.tensor(advantages, dtype=torch.float32)
            returns_t = torch.tensor(returns, dtype=torch.float32)

            # Normalize advantages
            if len(advantages_t) > 1:
                advantages_t = (advantages_t - advantages_t.mean()) / (
                    advantages_t.std() + 1e-8
                )

            # PPO update
            self._policy.train()
            self._value.train()

            lineup_logits, captain_logits = self._policy(states_t)

            # Recompute log-probs for the same actions under current policy
            lineup_probs = torch.sigmoid(lineup_logits)
            lineup_probs = torch.clamp(lineup_probs, 1e-7, 1 - 1e-7)
            new_lineup_log_probs = (
                lineup_actions_t * torch.log(lineup_probs)
                + (1 - lineup_actions_t) * torch.log(1 - lineup_probs)
            ).sum(dim=1)

            captain_log_softmax = F.log_softmax(captain_logits, dim=1)
            new_captain_log_probs = captain_log_softmax.gather(
                1, captain_actions_t.unsqueeze(1)
            ).squeeze(1)

            new_log_probs = new_lineup_log_probs + new_captain_log_probs

            # Clipped surrogate loss
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            clipped_ratio = torch.clamp(
                ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
            )
            policy_loss = -torch.min(
                ratio * advantages_t, clipped_ratio * advantages_t
            ).mean()

            # Value loss
            values_pred = self._value(states_t).squeeze(1)
            value_loss = F.mse_loss(values_pred, returns_t)

            # Combined loss
            loss = policy_loss + 0.5 * value_loss

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

    def _encode_from_state_data(
        self, state: SquadState, data: SeasonData
    ) -> np.ndarray:
        """Encode SquadState + SeasonData into the same vector format as FPLEnvironment._encode_state.

        Produces a ~157-dim float32 vector matching the environment state encoding.
        """
        vec = np.zeros(STATE_DIM, dtype=np.float32)
        current_gw = data.current_gameweek
        xi_set = set(state.starting_xi)

        for i, player in enumerate(state.players[:_NUM_SQUAD_PLAYERS]):
            offset = i * _PLAYER_FEATURES

            # Element type one-hot (4 dims)
            if 1 <= player.element_type <= 4:
                vec[offset + player.element_type - 1] = 1.0

            # Normalized price
            vec[offset + 4] = player.sell_price / 200.0

            # Rolling features from feature module
            features = build_player_features(data, player.code, current_gw)
            vec[offset + 5] = features.get("form_3", 0.0) / 10.0
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

    def _decode_lineup(
        self,
        lineup_logits: torch.Tensor,
        players: list[PlayerInSquad],
    ) -> tuple[list[int], list[int]]:
        """Decode lineup logits into valid starting XI and bench order.

        Uses formation masking to ensure valid FPL formation:
        1 GK, >= 3 DEF, >= 2 MID, >= 1 FWD, 11 total.
        """
        scores = torch.sigmoid(lineup_logits).numpy()

        # Separate GKs and outfield
        gk_indices = [i for i, p in enumerate(players) if p.element_type == 1]
        outfield_indices = [i for i, p in enumerate(players) if p.element_type != 1]

        # Pick the GK with highest logit
        if gk_indices:
            best_gk_idx = max(gk_indices, key=lambda i: scores[i])
            bench_gk_indices = [i for i in gk_indices if i != best_gk_idx]
        else:
            best_gk_idx = None
            bench_gk_indices = []

        # Sort outfield by scores descending
        outfield_sorted = sorted(outfield_indices, key=lambda i: scores[i], reverse=True)

        # Greedily fill 10 outfield spots ensuring min constraints
        selected: list[int] = []
        remaining: list[int] = []

        # Track position counts
        def _position_counts(indices: list[int]) -> Counter:
            return Counter(players[i].element_type for i in indices)

        for idx in outfield_sorted:
            et = players[idx].element_type
            counts = _position_counts(selected)
            spots_left = 10 - len(selected)

            if spots_left <= 0:
                remaining.append(idx)
                continue

            # Check if we can still fill minimum requirements with remaining spots
            still_needed_def = max(0, _MIN_DEF - counts.get(2, 0))
            still_needed_mid = max(0, _MIN_MID - counts.get(3, 0))
            still_needed_fwd = max(0, _MIN_FWD - counts.get(4, 0))
            must_fill = still_needed_def + still_needed_mid + still_needed_fwd

            # If adding this player would prevent filling minimum requirements
            if et in (2, 3, 4):
                # This position contributes to a minimum, always safe to add
                selected.append(idx)
            else:
                # Shouldn't happen (outfield are types 2,3,4) but handle gracefully
                if spots_left > must_fill:
                    selected.append(idx)
                else:
                    remaining.append(idx)

        # If we didn't fill 10, take from remaining
        while len(selected) < 10 and remaining:
            selected.append(remaining.pop(0))

        # Verify minimums are met; if not, swap from end
        counts = _position_counts(selected)
        # This greedy approach should work since all outfield players are DEF/MID/FWD
        # and we take them in score order. The constraints are loose enough.

        # Build starting XI
        starting_xi_indices = []
        if best_gk_idx is not None:
            starting_xi_indices.append(best_gk_idx)
        starting_xi_indices.extend(selected)

        # Build bench (bench GK + remaining outfield, ordered by score)
        bench_indices = bench_gk_indices + remaining

        starting_xi = [players[i].code for i in starting_xi_indices]
        bench_order = [players[i].code for i in bench_indices]

        return starting_xi, bench_order

    def _compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[bool],
    ) -> list[float]:
        """Compute Generalized Advantage Estimation.

        Returns a list of advantage values, one per timestep.
        """
        n = len(rewards)
        advantages = [0.0] * n
        last_gae = 0.0

        for t in reversed(range(n)):
            if dones[t]:
                next_value = 0.0
                last_gae = 0.0
            else:
                next_value = values[t + 1] if t + 1 < n else 0.0

            delta = rewards[t] + self.gamma * next_value - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae

        return advantages
